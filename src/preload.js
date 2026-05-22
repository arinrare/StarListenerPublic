const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  runPython: (text) => ipcRenderer.invoke('run-python', text),
  runTTS: (payload) => ipcRenderer.invoke('run-tts', payload),
  onTtsStatus: (callback) => ipcRenderer.on('tts-status', (_event, data) => callback(data)),
  openFile: () => ipcRenderer.invoke('open-file'),
  readJsonFile: (path) => ipcRenderer.invoke('read-json-file', path),
  writeJsonFile: (path, data) => ipcRenderer.invoke('write-json-file', path, data),
  resolvePath: (path) => ipcRenderer.invoke('resolve-path', path),
  saveAppData: (data) => ipcRenderer.invoke('save-app-data', data),
  loadAppData: () => ipcRenderer.invoke('load-app-data'),
  fileExists: (path) => ipcRenderer.invoke('file-exists', path)
});