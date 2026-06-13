import React, { useRef, useCallback, useState } from 'react';
import Webcam from 'react-webcam';

interface CameraCaptureProps {
    onCapture: (imageSrc: string) => void;
    onClose: () => void;
}

const videoConstraints = {
    width: 1280,
    height: 720,
    facingMode: "environment"
};

export const CameraCapture: React.FC<CameraCaptureProps> = ({ onCapture, onClose }) => {
    const webcamRef = useRef<Webcam>(null);
    const [isCapturing, setIsCapturing] = useState(false);

    const capture = useCallback(() => {
        setIsCapturing(true);
        const imageSrc = webcamRef.current?.getScreenshot();
        if (imageSrc) {
            onCapture(imageSrc);
        }
        setIsCapturing(false);
    }, [webcamRef, onCapture]);

    return (
        <div className="fixed inset-0 z-[10000] bg-black flex flex-col">
            <div className="relative flex-1 bg-black flex items-center justify-center overflow-hidden">
                <Webcam
                    audio={false}
                    ref={webcamRef}
                    screenshotFormat="image/jpeg"
                    videoConstraints={videoConstraints}
                    className="absolute inset-0 w-full h-full object-cover"
                    mirrored={false}
                />

                {/* Overlay Guides */}
                <div className="absolute inset-0 border-2 border-white/30 m-8 rounded-xl pointer-events-none">
                    <div className="absolute top-0 left-0 w-8 h-8 border-t-4 border-l-4 border-white rounded-tl-xl"></div>
                    <div className="absolute top-0 right-0 w-8 h-8 border-t-4 border-r-4 border-white rounded-tr-xl"></div>
                    <div className="absolute bottom-0 left-0 w-8 h-8 border-b-4 border-l-4 border-white rounded-bl-xl"></div>
                    <div className="absolute bottom-0 right-0 w-8 h-8 border-b-4 border-r-4 border-white rounded-br-xl"></div>
                </div>
            </div>

            {/* Controls */}
            <div className="h-32 bg-black/80 backdrop-blur-md flex items-center justify-between px-10 pb-8 pt-4">
                <button
                    onClick={onClose}
                    className="w-12 h-12 rounded-full bg-gray-800 flex items-center justify-center text-white hover:bg-gray-700 transition-colors"
                >
                    <span className="material-symbols-outlined">close</span>
                </button>

                <button
                    onClick={capture}
                    disabled={isCapturing}
                    className="w-20 h-20 rounded-full border-4 border-white flex items-center justify-center p-1"
                >
                    <div className={`w-full h-full bg-white rounded-full transition-transform ${isCapturing ? 'scale-90' : 'scale-100 hover:scale-95'}`}></div>
                </button>

                <button
                    className="w-12 h-12 rounded-full bg-gray-800 flex items-center justify-center text-white hover:bg-gray-700 transition-colors opacity-0 pointer-events-none"
                >
                    <span className="material-symbols-outlined">flip_camera_ios</span>
                </button>
            </div>
        </div>
    );
};
