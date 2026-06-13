// Shim for expo-image-picker that uses @capacitor/camera on Android
// Falls back to HTML file input on web

export const MediaTypeOptions = {
  All: 'All' as const,
  Images: 'Images' as const,
  Videos: 'Videos' as const,
};

interface ImagePickerResult {
  canceled: boolean;
  assets: Array<{
    uri: string;
    width: number;
    height: number;
    type?: string;
    fileName?: string;
  }>;
}

interface ImagePickerOptions {
  mediaTypes?: string;
  allowsEditing?: boolean;
  aspect?: [number, number];
  quality?: number;
  base64?: boolean;
}

async function tryCapacitorCamera(source: 'PHOTOS' | 'CAMERA'): Promise<ImagePickerResult | null> {
  try {
    const { Camera, CameraResultType, CameraSource } = await import('@capacitor/camera');
    const cameraSource = source === 'CAMERA' ? CameraSource.Camera : CameraSource.Photos;
    const image = await Camera.getPhoto({
      quality: 90,
      allowEditing: true,
      resultType: CameraResultType.Uri,
      source: cameraSource,
    });

    if (image.webPath) {
      return {
        canceled: false,
        assets: [{
          uri: image.webPath,
          width: 0,
          height: 0,
          type: `image/${image.format || 'jpeg'}`,
        }],
      };
    }
    return { canceled: true, assets: [] };
  } catch (e: any) {
    // User cancelled or plugin not available
    if (e?.message?.includes('User cancelled') || e?.message?.includes('cancel')) {
      return { canceled: true, assets: [] };
    }
    return null; // Signal to try fallback
  }
}

function htmlFileInput(): Promise<ImagePickerResult> {
  return new Promise((resolve) => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'image/*';
    input.onchange = (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (file) {
        const url = URL.createObjectURL(file);
        resolve({
          canceled: false,
          assets: [{
            uri: url,
            width: 0,
            height: 0,
            type: file.type,
            fileName: file.name,
          }],
        });
      } else {
        resolve({ canceled: true, assets: [] });
      }
    };
    input.oncancel = () => resolve({ canceled: true, assets: [] });
    input.click();
  });
}

export async function launchImageLibraryAsync(_options?: ImagePickerOptions): Promise<ImagePickerResult> {
  // Try Capacitor Camera (picks from gallery)
  const result = await tryCapacitorCamera('PHOTOS');
  if (result !== null) return result;
  // Fallback to HTML file input
  return htmlFileInput();
}

export async function launchCameraAsync(_options?: ImagePickerOptions): Promise<ImagePickerResult> {
  // Try Capacitor Camera
  const result = await tryCapacitorCamera('CAMERA');
  if (result !== null) return result;
  // Fallback to HTML file input with camera capture
  return new Promise((resolve) => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'image/*';
    input.capture = 'environment';
    input.onchange = (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (file) {
        const url = URL.createObjectURL(file);
        resolve({
          canceled: false,
          assets: [{
            uri: url,
            width: 0,
            height: 0,
            type: file.type,
            fileName: file.name,
          }],
        });
      } else {
        resolve({ canceled: true, assets: [] });
      }
    };
    input.oncancel = () => resolve({ canceled: true, assets: [] });
    input.click();
  });
}

export async function requestMediaLibraryPermissionsAsync() {
  return { status: 'granted', granted: true };
}

export async function requestCameraPermissionsAsync() {
  try {
    const { Camera } = await import('@capacitor/camera');
    const perms = await Camera.requestPermissions({ permissions: ['camera'] });
    return {
      status: perms.camera === 'granted' ? 'granted' : 'denied',
      granted: perms.camera === 'granted',
    };
  } catch {
    return { status: 'granted', granted: true };
  }
}

export default {
  MediaTypeOptions,
  launchImageLibraryAsync,
  launchCameraAsync,
  requestMediaLibraryPermissionsAsync,
  requestCameraPermissionsAsync,
};
