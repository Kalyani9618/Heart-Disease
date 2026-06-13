// Mock for expo-modules-core
// Provides missing exports that cause build failures in Vite

import { DeviceEventEmitter as RNDeviceEventEmitter } from 'react-native-web';

export const DeviceEventEmitter = RNDeviceEventEmitter;

export class EventEmitter {
    _listenerCount = 0;
    constructor(nativeModule?: any) { }
    addListener() { return { remove: () => { } }; }
    removeListeners() { }
    emit() { }
    removeAllListeners() { }
    listenerCount() { return 0; }
}

export const NativeModulesProxy = {};

export const Platform = {
    OS: 'web',
    select: (obj: any) => obj.web || obj.default,
};

export const requireNativeViewManager = (name: string) => {
    console.log('Mock requireNativeViewManager called for:', name);
    return null;
};

export class SharedObject { }

export class SharedRef extends SharedObject { }

export class NativeModule {
    addListener() { return { remove: () => { } }; }
    removeListeners() { }
}

export class CodedError extends Error {
    code: string;
    constructor(code: string, message: string) {
        super(message);
        this.code = code;
        this.name = 'CodedError';
    }
}

export class UnavailabilityError extends Error {
    constructor(moduleName: string, propertyName: string) {
        super(`The property ${propertyName} is unavailable in ${moduleName}`);
        this.name = 'UnavailabilityError';
    }
}

export const uuid = {
    v4: () => 'mock-uuid-v4',
};

export const SyntheticPlatformEmitter = DeviceEventEmitter;

export const registerWebModule = (module: any) => {
    console.log('Mock registerWebModule called for:', module);
};

export const createWebModule = (moduleImplementation: any) => {
    return moduleImplementation;
};

export const requireNativeModule = (name: string) => {
    console.log('Mock requireNativeModule called for:', name);
    return {};
};

export const requireOptionalNativeModule = (name: string) => {
    console.log('Mock requireOptionalNativeModule called for:', name);
    return null;
};

// Add other commonly used exports if needed
export const createPermissionHook = () => () => ({ status: 'granted', expires: 'never', canAskAgain: true, granted: true });
export const PermissionStatus = {
    GRANTED: 'granted',
    UNDETERMINED: 'undetermined',
    DENIED: 'denied',
};

export default {
    DeviceEventEmitter,
    EventEmitter,
    NativeModulesProxy,
    Platform,
    requireNativeViewManager,
    SharedObject,
    SharedRef,
    NativeModule,
    CodedError,
    UnavailabilityError,
    uuid,
    SyntheticPlatformEmitter,
    registerWebModule,
    createWebModule,
    requireNativeModule,
    requireOptionalNativeModule,
    createPermissionHook,
    PermissionStatus,
};
