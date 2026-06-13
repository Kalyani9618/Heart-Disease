// Mock for @capacitor/filesystem - web fallback
// On web, Filesystem operations are not supported; native code paths handle this via try/catch

export enum Directory {
  Documents = 'DOCUMENTS',
  Data = 'DATA',
  Library = 'LIBRARY',
  Cache = 'CACHE',
  External = 'EXTERNAL',
  ExternalStorage = 'EXTERNAL_STORAGE',
}

export enum Encoding {
  UTF8 = 'utf8',
  ASCII = 'ascii',
  UTF16 = 'utf16',
}

export const Filesystem = {
  async writeFile(_options: any): Promise<any> {
    console.warn('Filesystem.writeFile is not available on web');
    throw new Error('Filesystem not available on web');
  },
  async readFile(_options: any): Promise<any> {
    console.warn('Filesystem.readFile is not available on web');
    throw new Error('Filesystem not available on web');
  },
  async deleteFile(_options: any): Promise<void> {
    console.warn('Filesystem.deleteFile is not available on web');
  },
  async mkdir(_options: any): Promise<void> {
    console.warn('Filesystem.mkdir is not available on web');
  },
  async readdir(_options: any): Promise<any> {
    return { files: [] };
  },
  async getUri(_options: any): Promise<any> {
    return { uri: '' };
  },
  async stat(_options: any): Promise<any> {
    throw new Error('Filesystem not available on web');
  },
  async copy(_options: any): Promise<any> {
    throw new Error('Filesystem not available on web');
  },
};
