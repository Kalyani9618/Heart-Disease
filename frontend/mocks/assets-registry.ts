// Mock for @react-native/assets-registry/registry
// This is only used in native builds, so we provide a no-op for web

export function getAssetByID(_id: number): null {
    return null;
}

export function registerAsset(asset: { name: string; type: string }) {
    return asset;
}

export default {
    getAssetByID,
    registerAsset,
};
