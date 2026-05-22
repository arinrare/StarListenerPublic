const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path = require('path');
const fs = require('fs');
const os = require('os');
const { spawn } = require('child_process');

let _dotEnvLoaded = false;

function loadDotEnv(rootPath) {
  if (_dotEnvLoaded) return;
  _dotEnvLoaded = true;

  try {
    const envPath = path.join(rootPath, '.env');
    if (!fs.existsSync(envPath)) return;
    const content = fs.readFileSync(envPath, 'utf8');
    const lines = content.split(/\r?\n/);
    for (const line of lines) {
      const trimmed = String(line || '').trim();
      if (!trimmed || trimmed.startsWith('#')) continue;

      const eq = trimmed.indexOf('=');
      if (eq === -1) continue;
      const key = trimmed.slice(0, eq).trim().replace(/^export\s+/i, '');
      if (!key) continue;

      let value = trimmed.slice(eq + 1).trim();
      // Remove surrounding quotes.
      if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) {
        value = value.slice(1, -1);
      }

      // Only set if not already set in the environment.
      if (process.env[key] === undefined) {
        process.env[key] = value;
      }
    }
  } catch (e) {
    console.warn('Failed to load .env:', e && e.message ? e.message : e);
  }
}

function createWindow() {
  const win = new BrowserWindow({
    width: 1000,
    height: 800,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      nodeIntegration: false,
      contextIsolation: true,
      webSecurity: false
    }
  });

  win.loadFile('src/index.html');
}

function getPythonPaths() {
  const isDev = !app.isPackaged || process.env.NODE_ENV === 'development';
  const rootPath = app.getAppPath();

  let pythonExec;
  let scriptPath;
  
    if (isDev) {
    // Windows: .venv/Scripts/python.exe | Mac/Linux: .venv/bin/python
        pythonExec = process.platform === 'win32'
        ? path.join(rootPath, '.venv', 'Scripts', 'python.exe')
        : path.join(rootPath, '.venv', 'bin', 'python');

        if (!fs.existsSync(pythonExec)) {
            console.warn("VENV not found at " + pythonExec + ". Falling back to global 'python'.");
            pythonExec = 'python'; 
        }

        scriptPath = path.join(rootPath, 'engine', 'sl_engine.py');
        
    } else {
        pythonExec = path.join(process.resourcesPath, 'engine', 'engine.exe');
        scriptPath = path.join(process.resourcesPath, 'engine', 'sl_engine.py');
    }
    return { pythonExec, scriptPath };
}

ipcMain.handle('open-file', async () => {
  const { canceled, filePaths } = await dialog.showOpenDialog({
    properties: ['openFile'],
    filters: [{ name: 'Documents', extensions: ['epub', 'pdf', 'txt', 'md', 'html', 'htm', 'xhtml'] }]
  });
  if (!canceled) return filePaths[0];
});

ipcMain.handle('read-json-file', async (_event, filePath) => {
  const resolved = path.isAbsolute(filePath) ? filePath : path.join(app.getAppPath(), filePath);
  try {
    return JSON.parse(fs.readFileSync(resolved, 'utf8'));
  } catch (e) {
    if (e.code === 'ENOENT') return null;
    throw e;
  }
});

ipcMain.handle('resolve-path', async (_event, relPath) => {
  return path.resolve(app.getAppPath(), relPath);
});

ipcMain.handle('load-app-data', async () => {
  const resolved = path.join(app.getAppPath(), 'output', 'starlistener.json');
  if (fs.existsSync(resolved)) {
    return JSON.parse(fs.readFileSync(resolved, 'utf8'));
  }
  return null;
});

ipcMain.handle('file-exists', async (_event, filePath) => {
  const resolved = path.isAbsolute(filePath) ? filePath : path.join(app.getAppPath(), filePath);
  return fs.existsSync(resolved);
});

ipcMain.handle('save-app-data', async (_event, data) => {
  const resolved = path.join(app.getAppPath(), 'output', 'starlistener.json');
  const dir = path.dirname(resolved);
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(resolved, JSON.stringify(data, null, 2), 'utf8');
});

ipcMain.handle('write-json-file', async (_event, filePath, data) => {
  const resolved = path.isAbsolute(filePath) ? filePath : path.join(app.getAppPath(), filePath);
  const dir = path.dirname(resolved);
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(resolved, JSON.stringify(data, null, 2), 'utf8');
});

app.whenReady().then(() => {
  loadDotEnv(app.getAppPath());
  createWindow();
});

ipcMain.handle('run-tts', async (event, args) => {
  loadDotEnv(app.getAppPath());
  const { pythonExec } = getPythonPaths();
  const scriptPath = path.join(app.getAppPath(), 'engine', 'sl_tts.py');

  const tmpFile = path.join(os.tmpdir(), `starlistener_tts_${Date.now()}.json`);
  fs.writeFileSync(tmpFile, String(args || ''), 'utf8');

  return new Promise((resolve, reject) => {
    const pythonProcess = spawn(pythonExec, ['-u', scriptPath, '--file', tmpFile], {
      env: { ...process.env, PYTHONUNBUFFERED: '1' }
    });

    let stdoutData = '';
    let stderrData = '';
    let stderrPartial = '';

    pythonProcess.stdout.on('data', (data) => {
      stdoutData += data.toString();
    });
    pythonProcess.stderr.on('data', (data) => {
      const str = data.toString();
      stderrData += str;
      stderrPartial += str;
      const lines = stderrPartial.split(/\r?\n/);
      stderrPartial = lines.pop();
      for (const line of lines) {
        if (!line) continue;
        try {
          const parsed = JSON.parse(line);
          if (parsed && parsed.status) {
            console.log("[tts-status]", JSON.stringify(parsed));
            event.sender.send('tts-status', parsed);
            continue;
          }
        } catch (_) {}
        console.log("[py:tts:stderr]", line.slice(0, 200));
      }
    });

    pythonProcess.on('error', (err) => {
      try { fs.unlinkSync(tmpFile); } catch (_) {}
      reject(`Failed to start Python TTS process: ${err.message}`);
    });

    pythonProcess.on('close', (code) => {
      try { fs.unlinkSync(tmpFile); } catch (_) {}
      if (code === 0) {
        const cleaned = stdoutData.trim();
        const start = cleaned.indexOf('{');
        const end = cleaned.lastIndexOf('}');
        if (start !== -1 && end !== -1) {
          resolve(cleaned.slice(start, end + 1));
        } else {
          resolve(cleaned);
        }
      } else {
        reject(`Python TTS exited with code ${code}. Error: ${stderrData}`);
      }
    });
  });
});

ipcMain.handle('run-python', async (event, args) => {
  // Defensive: in case run-python is called before app.whenReady finishes.
  loadDotEnv(app.getAppPath());
  const { pythonExec, scriptPath } = getPythonPaths();

    return new Promise((resolve, reject) => {   
    const pythonProcess = spawn(pythonExec, [scriptPath, args]);

    let stdoutData = '';
    let stderrData = '';

    pythonProcess.stdout.on('data', (data) => {
        const str = data.toString();
        console.log("Python stdout:", str);
        stdoutData += str;
    });
    pythonProcess.stderr.on('data', (data) => {
        const str = data.toString();
        // Preserve multi-line stderr and keep it readable.
        for (const line of str.split(/\r?\n/)) {
          if (!line) continue;
          console.log("[py:stderr]", line);
        }
        stderrData += str;
    });

    pythonProcess.on('error', (err) => {
      reject(`Failed to start Python process: ${err.message}`);
    });

    pythonProcess.on('close', (code) => {
      if (code === 0) {
        resolve(stdoutData.trim());
      } else {
        // If the process failed, we reject with the accumulated stderr
        reject(`Python exited with code ${code}. Error: ${stderrData}`);
      }
    });
  });
});