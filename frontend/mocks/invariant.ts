// Mock for invariant module
// The original invariant package uses CommonJS exports, but some dependencies expect ESM default export
// This shim provides a compatible default export for web builds

function invariant(condition: any, message?: string): asserts condition {
    if (!condition) {
        throw new Error(message || 'Invariant violation');
    }
}

export default invariant;
export { invariant };
