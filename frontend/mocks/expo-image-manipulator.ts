/**
 * Web-compatible mock for expo-image-manipulator
 *
 * Uses the Canvas API to provide manipulateAsync and SaveFormat
 * so the DocumentScanner and image compression utilities work on web
 * without pulling in native-only Expo dependencies (SharedRef, etc.).
 */

export enum SaveFormat {
  JPEG = 'jpeg',
  PNG = 'png',
  WEBP = 'webp',
}

export enum FlipType {
  Horizontal = 'horizontal',
  Vertical = 'vertical',
}

export interface Action {
  resize?: { width?: number; height?: number };
  rotate?: number;
  flip?: FlipType;
  crop?: { originX: number; originY: number; width: number; height: number };
}

export interface ImageResult {
  uri: string;
  width: number;
  height: number;
  base64?: string;
}

export interface ManipulateOptions {
  compress?: number;
  format?: SaveFormat;
  base64?: boolean;
}

/**
 * Load an image from a URI and return an HTMLImageElement.
 */
function loadImage(uri: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => resolve(img);
    img.onerror = (e) => reject(new Error(`Failed to load image: ${e}`));
    img.src = uri;
  });
}

/**
 * Web implementation of expo-image-manipulator's manipulateAsync.
 * Supports resize, rotate, flip, and crop actions via Canvas API.
 */
export async function manipulateAsync(
  uri: string,
  actions: Action[] = [],
  options: ManipulateOptions = {}
): Promise<ImageResult> {
  const { compress = 0.8, format = SaveFormat.JPEG, base64: includeBase64 = false } = options;

  const img = await loadImage(uri);
  let width = img.naturalWidth;
  let height = img.naturalHeight;

  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('Canvas 2D context not available');

  // Apply actions sequentially
  // First pass: calculate final dimensions
  let currentWidth = width;
  let currentHeight = height;

  for (const action of actions) {
    if (action.resize) {
      if (action.resize.width && action.resize.height) {
        currentWidth = action.resize.width;
        currentHeight = action.resize.height;
      } else if (action.resize.width) {
        const ratio = action.resize.width / currentWidth;
        currentWidth = action.resize.width;
        currentHeight = Math.round(currentHeight * ratio);
      } else if (action.resize.height) {
        const ratio = action.resize.height / currentHeight;
        currentHeight = Math.round(currentWidth * ratio);
        currentWidth = Math.round(currentWidth * ratio);
      }
    }
    if (action.rotate && (action.rotate === 90 || action.rotate === 270 || action.rotate === -90)) {
      [currentWidth, currentHeight] = [currentHeight, currentWidth];
    }
    if (action.crop) {
      currentWidth = action.crop.width;
      currentHeight = action.crop.height;
    }
  }

  canvas.width = currentWidth;
  canvas.height = currentHeight;

  // Draw the original image
  let sourceCanvas = document.createElement('canvas');
  let sourceCtx = sourceCanvas.getContext('2d')!;
  sourceCanvas.width = img.naturalWidth;
  sourceCanvas.height = img.naturalHeight;
  sourceCtx.drawImage(img, 0, 0);

  // Apply each action
  for (const action of actions) {
    if (action.crop) {
      const cropCanvas = document.createElement('canvas');
      const cropCtx = cropCanvas.getContext('2d')!;
      cropCanvas.width = action.crop.width;
      cropCanvas.height = action.crop.height;
      cropCtx.drawImage(
        sourceCanvas,
        action.crop.originX, action.crop.originY,
        action.crop.width, action.crop.height,
        0, 0,
        action.crop.width, action.crop.height
      );
      sourceCanvas = cropCanvas;
      sourceCtx = cropCtx;
    }

    if (action.resize) {
      let newW = sourceCanvas.width;
      let newH = sourceCanvas.height;
      if (action.resize.width && action.resize.height) {
        newW = action.resize.width;
        newH = action.resize.height;
      } else if (action.resize.width) {
        const ratio = action.resize.width / newW;
        newW = action.resize.width;
        newH = Math.round(newH * ratio);
      } else if (action.resize.height) {
        const ratio = action.resize.height / newH;
        newH = action.resize.height;
        newW = Math.round(newW * ratio);
      }
      const resizeCanvas = document.createElement('canvas');
      const resizeCtx = resizeCanvas.getContext('2d')!;
      resizeCanvas.width = newW;
      resizeCanvas.height = newH;
      resizeCtx.drawImage(sourceCanvas, 0, 0, newW, newH);
      sourceCanvas = resizeCanvas;
      sourceCtx = resizeCtx;
    }

    if (action.rotate) {
      const rad = (action.rotate * Math.PI) / 180;
      const sin = Math.abs(Math.sin(rad));
      const cos = Math.abs(Math.cos(rad));
      const newW = Math.round(sourceCanvas.width * cos + sourceCanvas.height * sin);
      const newH = Math.round(sourceCanvas.width * sin + sourceCanvas.height * cos);
      const rotateCanvas = document.createElement('canvas');
      const rotateCtx = rotateCanvas.getContext('2d')!;
      rotateCanvas.width = newW;
      rotateCanvas.height = newH;
      rotateCtx.translate(newW / 2, newH / 2);
      rotateCtx.rotate(rad);
      rotateCtx.drawImage(sourceCanvas, -sourceCanvas.width / 2, -sourceCanvas.height / 2);
      sourceCanvas = rotateCanvas;
      sourceCtx = rotateCtx;
    }

    if (action.flip) {
      const flipCanvas = document.createElement('canvas');
      const flipCtx = flipCanvas.getContext('2d')!;
      flipCanvas.width = sourceCanvas.width;
      flipCanvas.height = sourceCanvas.height;
      if (action.flip === FlipType.Horizontal) {
        flipCtx.scale(-1, 1);
        flipCtx.drawImage(sourceCanvas, -sourceCanvas.width, 0);
      } else {
        flipCtx.scale(1, -1);
        flipCtx.drawImage(sourceCanvas, 0, -sourceCanvas.height);
      }
      sourceCanvas = flipCanvas;
      sourceCtx = flipCtx;
    }
  }

  // Draw final result onto output canvas
  canvas.width = sourceCanvas.width;
  canvas.height = sourceCanvas.height;
  ctx.drawImage(sourceCanvas, 0, 0);

  // Convert to the desired format
  let mimeType = 'image/jpeg';
  if (format === SaveFormat.PNG) mimeType = 'image/png';
  else if (format === SaveFormat.WEBP) mimeType = 'image/webp';

  const quality = format === SaveFormat.PNG ? undefined : compress;

  const result: ImageResult = {
    uri: canvas.toDataURL(mimeType, quality),
    width: canvas.width,
    height: canvas.height,
  };

  if (includeBase64) {
    result.base64 = result.uri.split(',')[1];
  }

  return result;
}

// Default export for compatibility
export default {
  manipulateAsync,
  SaveFormat,
  FlipType,
};
