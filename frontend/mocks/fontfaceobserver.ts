// Mock for fontfaceobserver module
// The original package uses CommonJS exports, but expo-font expects ESM default export
// This shim provides a compatible FontFaceObserver for web builds

class FontFaceObserver {
    family: string;

    constructor(family: string, _descriptors?: object) {
        this.family = family;
    }

    load(_text?: string, _timeout?: number): Promise<void> {
        // In web, fonts are typically handled by CSS
        // This mock just resolves immediately
        return Promise.resolve();
    }
}

export default FontFaceObserver;
export { FontFaceObserver };
