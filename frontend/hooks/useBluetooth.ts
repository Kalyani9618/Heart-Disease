import { useEffect, useState, useCallback, useRef } from 'react';
import { BleClient, numberToUUID } from '@capacitor-community/bluetooth-le';

// ============================================================================
// Standard BLE Service & Characteristic UUIDs for Health Devices
// ============================================================================

export const BLE_SERVICES = {
  HEART_RATE: numberToUUID(0x180d),
  BATTERY: numberToUUID(0x180f),
  DEVICE_INFO: numberToUUID(0x180a),
  BLOOD_PRESSURE: numberToUUID(0x1810),
  HEALTH_THERMOMETER: numberToUUID(0x1809),
  BODY_COMPOSITION: numberToUUID(0x181b),
  WEIGHT_SCALE: numberToUUID(0x181d),
  RUNNING_SPEED: numberToUUID(0x1814),
  CYCLING_SPEED: numberToUUID(0x1816),
};

export const BLE_CHARACTERISTICS = {
  HEART_RATE_MEASUREMENT: numberToUUID(0x2a37),
  BODY_SENSOR_LOCATION: numberToUUID(0x2a38),
  BATTERY_LEVEL: numberToUUID(0x2a19),
  MANUFACTURER_NAME: numberToUUID(0x2a29),
  MODEL_NUMBER: numberToUUID(0x2a24),
  FIRMWARE_REVISION: numberToUUID(0x2a26),
  BLOOD_PRESSURE_MEASUREMENT: numberToUUID(0x2a35),
  TEMPERATURE_MEASUREMENT: numberToUUID(0x2a1c),
};

// ============================================================================
// Types
// ============================================================================

export interface BluetoothDevice {
  deviceId: string;
  name: string;
  rssi: number;
  services?: string[];
}

export interface SmartwatchData {
  heartRate: number | null;
  heartRateTimestamp: string | null;
  batteryLevel: number | null;
  sensorLocation: string | null;
  manufacturerName: string | null;
  modelNumber: string | null;
  bloodPressureSystolic: number | null;
  bloodPressureDiastolic: number | null;
  bloodPressureTimestamp: string | null;
  temperature: number | null;
  temperatureTimestamp: string | null;
}

export interface UseBluetoothReturn {
  isInitialized: boolean;
  isScanning: boolean;
  isConnected: boolean;
  connectedDeviceId: string | null;
  devices: BluetoothDevice[];
  smartwatchData: SmartwatchData;
  error: string | null;
  startScan: () => Promise<void>;
  stopScan: () => Promise<void>;
  connectToDevice: (deviceId: string) => Promise<void>;
  disconnectDevice: (deviceId: string) => Promise<void>;
  readBatteryLevel: (deviceId: string) => Promise<number | null>;
  startHeartRateNotifications: (deviceId: string) => Promise<void>;
  stopHeartRateNotifications: (deviceId: string) => Promise<void>;
  startBloodPressureNotifications: (deviceId: string) => Promise<void>;
  stopBloodPressureNotifications: (deviceId: string) => Promise<void>;
  readTemperature: (deviceId: string) => Promise<number | null>;
  startAllMonitoring: (deviceId: string) => Promise<void>;
  stopAllMonitoring: (deviceId: string) => Promise<void>;
}

// ============================================================================
// Helper: Parse Heart Rate Measurement characteristic value
// ============================================================================

function parseHeartRate(value: DataView): number {
  const flags = value.getUint8(0);
  const is16bit = (flags & 0x01) !== 0;

  if (is16bit) {
    return value.getUint16(1, true);
  }
  return value.getUint8(1);
}

function parseSensorLocation(value: number): string {
  const locations: Record<number, string> = {
    0: 'Other',
    1: 'Chest',
    2: 'Wrist',
    3: 'Finger',
    4: 'Hand',
    5: 'Ear Lobe',
    6: 'Foot',
  };
  return locations[value] || 'Unknown';
}

/**
 * Decode an IEEE 11073 SFLOAT (16-bit) value.
 * Format: 4-bit exponent (signed) | 12-bit mantissa (signed)
 * Special values: NaN (0x07FF), NRes (0x0800), +INF (0x07FE), -INF (0x0802), Reserved (0x0801)
 */
function decodeSFLOAT(raw: number): number | null {
  const SPECIAL = [0x07FF, 0x0800, 0x07FE, 0x0802, 0x0801];
  if (SPECIAL.includes(raw)) return null;
  let exponent = (raw >> 12) & 0x0F;
  let mantissa = raw & 0x0FFF;
  // Sign-extend exponent (4-bit signed)
  if (exponent >= 0x08) exponent -= 0x10;
  // Sign-extend mantissa (12-bit signed)
  if (mantissa >= 0x0800) mantissa -= 0x1000;
  return mantissa * Math.pow(10, exponent);
}

/**
 * Parse Blood Pressure Measurement characteristic (0x2A35)
 */
function parseBloodPressure(value: DataView): { systolic: number; diastolic: number } {
  // Flags byte determines unit: 0 = mmHg, 1 = kPa
  // const flags = value.getUint8(0);
  // const isKPa = (flags & 0x01) !== 0;
  // SFLOAT values at offsets 1 and 3
  const systolicRaw = value.getUint16(1, true);
  const diastolicRaw = value.getUint16(3, true);
  const systolic = decodeSFLOAT(systolicRaw);
  const diastolic = decodeSFLOAT(diastolicRaw);
  return {
    systolic: systolic != null ? Math.round(systolic) : 0,
    diastolic: diastolic != null ? Math.round(diastolic) : 0,
  };
}

/**
 * Decode an IEEE 11073 FLOAT (32-bit) value.
 * Format: 8-bit exponent (signed) | 24-bit mantissa (signed)
 */
function decodeFLOAT(value: DataView, offset: number): number | null {
  const raw = value.getUint32(offset, true);
  const SPECIAL_MANTISSA = [0x007FFFFF, 0x00800000, 0x007FFFFE, 0x00800002, 0x00800001];
  let exponent = (raw >> 24) & 0xFF;
  let mantissa = raw & 0x00FFFFFF;
  if (SPECIAL_MANTISSA.includes(mantissa)) return null;
  // Sign-extend exponent (8-bit signed)
  if (exponent >= 0x80) exponent -= 0x100;
  // Sign-extend mantissa (24-bit signed)
  if (mantissa >= 0x800000) mantissa -= 0x1000000;
  return mantissa * Math.pow(10, exponent);
}

/**
 * Parse Temperature Measurement characteristic (0x2A1C)
 */
function parseTemperature(value: DataView): number {
  const flags = value.getUint8(0);
  const isFahrenheit = (flags & 0x01) !== 0;
  // IEEE 11073 FLOAT at offset 1 (4 bytes)
  const temp = decodeFLOAT(value, 1);
  if (temp == null) return 0;
  if (isFahrenheit) {
    return Math.round(((temp - 32) * 5 / 9) * 10) / 10;
  }
  return Math.round(temp * 10) / 10;
}

// ============================================================================
// Hook
// ============================================================================

export const useBluetooth = (): UseBluetoothReturn => {
  const [isInitialized, setIsInitialized] = useState(false);
  const [isScanning, setIsScanning] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [connectedDeviceId, setConnectedDeviceId] = useState<string | null>(null);
  const [devices, setDevices] = useState<BluetoothDevice[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [smartwatchData, setSmartwatchData] = useState<SmartwatchData>({
    heartRate: null,
    heartRateTimestamp: null,
    batteryLevel: null,
    sensorLocation: null,
    manufacturerName: null,
    modelNumber: null,
    bloodPressureSystolic: null,
    bloodPressureDiastolic: null,
    bloodPressureTimestamp: null,
    temperature: null,
    temperatureTimestamp: null,
  });

  const isScanningRef = useRef(false);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Initialize Bluetooth on mount
  useEffect(() => {
    const initBluetooth = async () => {
      try {
        await BleClient.initialize({ androidNeverForLocation: true });
        setIsInitialized(true);
        console.log('✓ Bluetooth initialized');
      } catch (err) {
        const errorMsg = err instanceof Error ? err.message : String(err);
        console.error('✗ Bluetooth initialization failed:', errorMsg);
        setError(errorMsg);
      }
    };

    initBluetooth();

    return () => {
      if (isScanningRef.current) {
        BleClient.stopLEScan().catch(() => {});
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
    };
  }, []);

  const startScan = useCallback(async () => {
    if (!isInitialized) {
      setError('Bluetooth not initialized. Please enable Bluetooth.');
      return;
    }

    try {
      setIsScanning(true);
      isScanningRef.current = true;
      setDevices([]);
      setError(null);

      await BleClient.requestLEScan(
        {
          services: [], // Empty scans all services; health devices will be found
          allowDuplicates: false,
        },
        (result) => {
          console.log('Device found:', result.device.deviceId, result.device.name);
          setDevices((prev) => {
            const existing = prev.find((d) => d.deviceId === result.device.deviceId);
            if (existing) {
              return prev.map((d) =>
                d.deviceId === result.device.deviceId
                  ? { ...d, rssi: result.rssi ?? d.rssi }
                  : d
              );
            }
            return [
              ...prev,
              {
                deviceId: result.device.deviceId,
                name: result.device.name || 'Unknown Device',
                rssi: result.rssi ?? -100,
                services: result.uuids,
              },
            ];
          });
        }
      );

      console.log('✓ Bluetooth scan started');

      // Auto-stop scan after 15 seconds
      setTimeout(async () => {
        if (isScanningRef.current) {
          await stopScan();
        }
      }, 15000);
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : String(err);
      console.error('✗ Bluetooth scan failed:', errorMsg);

      if (errorMsg.toLowerCase().includes('bluetooth')) {
        setError('Bluetooth is not enabled. Please turn on Bluetooth in your device settings.');
      } else if (errorMsg.toLowerCase().includes('permission')) {
        setError('Bluetooth permission denied. Please allow Bluetooth access in settings.');
      } else {
        setError(errorMsg);
      }
      setIsScanning(false);
      isScanningRef.current = false;
    }
  }, [isInitialized]);

  const stopScan = useCallback(async () => {
    try {
      await BleClient.stopLEScan();
      setIsScanning(false);
      isScanningRef.current = false;
      console.log('✓ Bluetooth scan stopped');
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : String(err);
      console.error('✗ Stop scan failed:', errorMsg);
      setError(errorMsg);
    }
  }, []);

  const connectToDevice = useCallback(async (deviceId: string) => {
    try {
      setError(null);

      await BleClient.connect(deviceId, (disconnectedDeviceId) => {
        console.log(`⚡ Device disconnected: ${disconnectedDeviceId}`);
        setIsConnected(false);
        setConnectedDeviceId(null);

        // Auto-reconnect after 3 seconds
        reconnectTimeoutRef.current = setTimeout(async () => {
          console.log(`🔄 Attempting reconnect to ${disconnectedDeviceId}...`);
          try {
            await BleClient.connect(disconnectedDeviceId);
            setIsConnected(true);
            setConnectedDeviceId(disconnectedDeviceId);
            console.log(`✓ Reconnected to ${disconnectedDeviceId}`);
          } catch {
            console.log('✗ Reconnection failed');
          }
        }, 3000);
      });

      setIsConnected(true);
      setConnectedDeviceId(deviceId);
      console.log(`✓ Connected to device: ${deviceId}`);

      // Try to read device info after connecting
      try {
        await readDeviceInfo(deviceId);
      } catch {
        // Non-critical
      }
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : String(err);
      console.error('✗ Connection failed:', errorMsg);
      setError(`Connection failed: ${errorMsg}`);
      throw err;
    }
  }, []);

  const disconnectDevice = useCallback(async (deviceId: string) => {
    try {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = null;
      }

      try {
        await BleClient.stopNotifications(
          deviceId,
          BLE_SERVICES.HEART_RATE,
          BLE_CHARACTERISTICS.HEART_RATE_MEASUREMENT
        );
      } catch {
        // May not have been started
      }

      await BleClient.disconnect(deviceId);
      setIsConnected(false);
      setConnectedDeviceId(null);
      setSmartwatchData({
        heartRate: null,
        heartRateTimestamp: null,
        batteryLevel: null,
        sensorLocation: null,
        manufacturerName: null,
        modelNumber: null,
        bloodPressureSystolic: null,
        bloodPressureDiastolic: null,
        bloodPressureTimestamp: null,
        temperature: null,
        temperatureTimestamp: null,
      });
      console.log(`✓ Disconnected from device: ${deviceId}`);
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : String(err);
      console.error('✗ Disconnect failed:', errorMsg);
      setError(errorMsg);
    }
  }, []);

  const readBatteryLevel = useCallback(async (deviceId: string): Promise<number | null> => {
    try {
      const result = await BleClient.read(
        deviceId,
        BLE_SERVICES.BATTERY,
        BLE_CHARACTERISTICS.BATTERY_LEVEL
      );
      const level = new DataView(result.buffer).getUint8(0);
      setSmartwatchData(prev => ({ ...prev, batteryLevel: level }));
      console.log(`🔋 Battery level: ${level}%`);
      return level;
    } catch (err) {
      console.warn('Could not read battery level:', err);
      return null;
    }
  }, []);

  const readDeviceInfo = useCallback(async (deviceId: string) => {
    try {
      const mfgResult = await BleClient.read(
        deviceId,
        BLE_SERVICES.DEVICE_INFO,
        BLE_CHARACTERISTICS.MANUFACTURER_NAME
      );
      const manufacturerName = new TextDecoder().decode(mfgResult);

      const modelResult = await BleClient.read(
        deviceId,
        BLE_SERVICES.DEVICE_INFO,
        BLE_CHARACTERISTICS.MODEL_NUMBER
      );
      const modelNumber = new TextDecoder().decode(modelResult);

      setSmartwatchData(prev => ({ ...prev, manufacturerName, modelNumber }));
      console.log(`📱 Device: ${manufacturerName} ${modelNumber}`);
    } catch {
      // Device info service is optional
    }
  }, []);

  const startHeartRateNotifications = useCallback(async (deviceId: string) => {
    try {
      try {
        const locResult = await BleClient.read(
          deviceId,
          BLE_SERVICES.HEART_RATE,
          BLE_CHARACTERISTICS.BODY_SENSOR_LOCATION
        );
        const location = parseSensorLocation(new DataView(locResult.buffer).getUint8(0));
        setSmartwatchData(prev => ({ ...prev, sensorLocation: location }));
      } catch {
        // Sensor location is optional
      }

      await BleClient.startNotifications(
        deviceId,
        BLE_SERVICES.HEART_RATE,
        BLE_CHARACTERISTICS.HEART_RATE_MEASUREMENT,
        (value) => {
          const heartRate = parseHeartRate(value);
          const timestamp = new Date().toISOString();
          setSmartwatchData(prev => ({
            ...prev,
            heartRate,
            heartRateTimestamp: timestamp,
          }));
          console.log(`❤️ Heart Rate: ${heartRate} BPM`);
        }
      );

      console.log('✓ Heart rate notifications started');
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : String(err);
      console.error('✗ Heart rate notifications failed:', errorMsg);
      setError(`Heart rate monitoring failed: ${errorMsg}`);
    }
  }, []);

  const stopHeartRateNotifications = useCallback(async (deviceId: string) => {
    try {
      await BleClient.stopNotifications(
        deviceId,
        BLE_SERVICES.HEART_RATE,
        BLE_CHARACTERISTICS.HEART_RATE_MEASUREMENT
      );
      console.log('✓ Heart rate notifications stopped');
    } catch (err) {
      console.error('✗ Stop HR notifications failed:', err);
    }
  }, []);

  const startBloodPressureNotifications = useCallback(async (deviceId: string) => {
    try {
      await BleClient.startNotifications(
        deviceId,
        BLE_SERVICES.BLOOD_PRESSURE,
        BLE_CHARACTERISTICS.BLOOD_PRESSURE_MEASUREMENT,
        (value) => {
          const bp = parseBloodPressure(value);
          const timestamp = new Date().toISOString();
          setSmartwatchData(prev => ({
            ...prev,
            bloodPressureSystolic: bp.systolic,
            bloodPressureDiastolic: bp.diastolic,
            bloodPressureTimestamp: timestamp,
          }));
          console.log(`🩸 Blood Pressure: ${bp.systolic}/${bp.diastolic} mmHg`);
        }
      );
      console.log('✓ Blood pressure notifications started');
    } catch (err) {
      console.warn('Blood pressure service not available on this device');
    }
  }, []);

  const stopBloodPressureNotifications = useCallback(async (deviceId: string) => {
    try {
      await BleClient.stopNotifications(
        deviceId,
        BLE_SERVICES.BLOOD_PRESSURE,
        BLE_CHARACTERISTICS.BLOOD_PRESSURE_MEASUREMENT
      );
      console.log('✓ Blood pressure notifications stopped');
    } catch (err) {
      // May not have been started
    }
  }, []);

  const readTemperature = useCallback(async (deviceId: string): Promise<number | null> => {
    try {
      // Temperature Measurement (0x2A1C) is typically indicate-only.
      // Subscribe to indications and resolve on the first value received.
      return await new Promise<number | null>((resolve) => {
        let resolved = false;
        const timeout = setTimeout(() => {
          if (!resolved) { resolved = true; resolve(null); }
        }, 10000);

        BleClient.startNotifications(
          deviceId,
          BLE_SERVICES.HEALTH_THERMOMETER,
          BLE_CHARACTERISTICS.TEMPERATURE_MEASUREMENT,
          (value) => {
            if (resolved) return;
            resolved = true;
            clearTimeout(timeout);
            const temp = parseTemperature(new DataView(value.buffer));
            const timestamp = new Date().toISOString();
            setSmartwatchData(prev => ({
              ...prev,
              temperature: temp,
              temperatureTimestamp: timestamp,
            }));
            console.log(`🌡️ Temperature: ${temp}°C`);
            // Stop indications after first reading
            BleClient.stopNotifications(
              deviceId,
              BLE_SERVICES.HEALTH_THERMOMETER,
              BLE_CHARACTERISTICS.TEMPERATURE_MEASUREMENT
            ).catch(() => {});
            resolve(temp);
          }
        ).catch((err) => {
          if (!resolved) { resolved = true; clearTimeout(timeout); resolve(null); }
        });
      });
    } catch (err) {
      console.warn('Temperature service not available on this device');
      return null;
    }
  }, []);

  const startAllMonitoring = useCallback(async (deviceId: string) => {
    console.log('🚀 Starting all vital monitoring...');
    await startHeartRateNotifications(deviceId);
    await startBloodPressureNotifications(deviceId);
    await readBatteryLevel(deviceId);
    await readTemperature(deviceId);
    console.log('✓ All monitoring started');
  }, [startHeartRateNotifications, startBloodPressureNotifications, readBatteryLevel, readTemperature]);

  const stopAllMonitoring = useCallback(async (deviceId: string) => {
    await stopHeartRateNotifications(deviceId);
    await stopBloodPressureNotifications(deviceId);
    console.log('✓ All monitoring stopped');
  }, [stopHeartRateNotifications, stopBloodPressureNotifications]);

  return {
    isInitialized,
    isScanning,
    isConnected,
    connectedDeviceId,
    devices,
    smartwatchData,
    error,
    startScan,
    stopScan,
    connectToDevice,
    disconnectDevice,
    readBatteryLevel,
    startHeartRateNotifications,
    stopHeartRateNotifications,
    startBloodPressureNotifications,
    stopBloodPressureNotifications,
    readTemperature,
    startAllMonitoring,
    stopAllMonitoring,
  };
};
