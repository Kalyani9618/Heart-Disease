import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { apiClient } from '../services/apiClient';
import { useBluetooth } from '../hooks/useBluetooth';
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    AreaChart,
    Area
} from 'recharts';

interface VitalData {
    time: string;
    value: number;
}

export default function SmartWatchScreen() {
    const navigate = useNavigate();
    const [loading, setLoading] = useState(false);
    const [vitals, setVitals] = useState<any>(null);
    const [deviceStatus, setDeviceStatus] = useState('Connected');
    const [isLive, setIsLive] = useState(false);
    const [heartRateHistory, setHeartRateHistory] = useState<VitalData[]>([]);

    // BLE integration via useBluetooth hook
    const {
        isInitialized: bleInitialized,
        isConnected: bleConnected,
        connectedDeviceId,
        smartwatchData,
        startAllMonitoring,
        stopAllMonitoring,
        readTemperature,
    } = useBluetooth();

    // When BLE is connected, start all monitoring
    useEffect(() => {
        if (bleConnected && connectedDeviceId) {
            setDeviceStatus('Connected (BLE)');
            setIsLive(true);
            startAllMonitoring(connectedDeviceId).catch(console.error);

            // Poll temperature every 30 seconds (thermometer may not support notifications)
            const tempInterval = setInterval(() => {
                if (connectedDeviceId) {
                    readTemperature(connectedDeviceId).catch(() => {});
                }
            }, 30000);

            return () => {
                stopAllMonitoring(connectedDeviceId).catch(console.error);
                clearInterval(tempInterval);
            };
        }
    }, [bleConnected, connectedDeviceId]);

    // Update vitals and chart from BLE smartwatch data
    useEffect(() => {
        if (bleConnected) {
            setVitals((prev: any) => ({
                ...prev,
                ...(smartwatchData.heartRate != null && { heart_rate: smartwatchData.heartRate }),
                ...(smartwatchData.bloodPressureSystolic != null && {
                    blood_pressure: `${smartwatchData.bloodPressureSystolic}/${smartwatchData.bloodPressureDiastolic}`,
                    bp_systolic: smartwatchData.bloodPressureSystolic,
                    bp_diastolic: smartwatchData.bloodPressureDiastolic,
                }),
                ...(smartwatchData.temperature != null && { temperature: smartwatchData.temperature }),
                ...(smartwatchData.batteryLevel != null && { battery: smartwatchData.batteryLevel }),
            }));

            if (smartwatchData.heartRate != null && smartwatchData.heartRate !== 0) {
                setHeartRateHistory(prev => {
                    const newPoint = {
                        time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
                        value: smartwatchData.heartRate!,
                    };
                    return [...prev, newPoint].slice(-30);
                });
            }
        }
    }, [bleConnected, smartwatchData.heartRate, smartwatchData.heartRateTimestamp, smartwatchData.bloodPressureTimestamp, smartwatchData.temperatureTimestamp]);

    useEffect(() => {
        const init = async () => {
            await loadVitals();
        };
        init();
    }, []);

    useEffect(() => {
        // Skip WebSocket fallback if BLE is providing real data
        if (bleConnected) return;

        // Get device ID from storage or default
        const devicesRaw = localStorage.getItem('connected_devices');
        let deviceId = 'mock_device_id';
        if (devicesRaw) {
            const devices = JSON.parse(devicesRaw);
            const activeWatch = devices.find((d: any) => d.status === 'connected' && d.type === 'watch');
            if (activeWatch) {
                deviceId = activeWatch.id;
            }
        }

        // WebSocket connection
        const wsUrl = apiClient.getWebSocketUrl(`/api/smartwatch/ws/${deviceId}`);
        const ws = new WebSocket(wsUrl);

        ws.onopen = () => {
            console.log('Connected to smartwatch stream');
            setIsLive(true);
        };

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                if (data.type === 'vitals') {
                    setVitals((prev: any) => ({
                        ...prev,
                        ...data.payload
                    }));

                    // Update chart history
                    if (data.payload.heart_rate) {
                        setHeartRateHistory(prev => {
                            const newPoint = {
                                time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
                                value: data.payload.heart_rate
                            };
                            const newHistory = [...prev, newPoint];
                            // Keep last 20 points
                            return newHistory.slice(-20);
                        });
                    }
                }
            } catch (e) {
                console.error('WS parse error:', e);
            }
        };

        ws.onclose = () => {
            setIsLive(false);
        };

        return () => {
            ws.close();
        };
    }, [bleConnected]);

    const loadVitals = async () => {
        setLoading(true);
        try {
            // Get device ID from storage
            const devicesRaw = localStorage.getItem('connected_devices');
            let deviceId = 'mock_device_id';

            if (devicesRaw) {
                const devices = JSON.parse(devicesRaw);
                const activeWatch = devices.find((d: any) => d.status === 'connected' && d.type === 'watch');
                if (activeWatch) {
                    deviceId = activeWatch.id;
                    setDeviceStatus('Connected');
                } else {
                    setDeviceStatus('Disconnected');
                }
            }

            // Fetch metrics in parallel
            const [hrData, stepsData, spo2Data] = await Promise.all([
                apiClient.getAggregatedVitals(deviceId, 'hr', 'day'),
                apiClient.getAggregatedVitals(deviceId, 'steps', 'day'),
                apiClient.getAggregatedVitals(deviceId, 'spo2', 'day'),
            ]);

            // Helper to get latest value
            const getValue = (data: any) => {
                if (data && data.data && data.data.length > 0) {
                    return data.data[data.data.length - 1].value;
                }
                return 0;
            };

            setVitals({
                heart_rate: Math.round(getValue(hrData)),
                steps: Math.round(getValue(stepsData)),
                calories: 0,
                sleep: '0h 0m',
                spo2: Math.round(getValue(spo2Data)),
            });

            // Initialize history with some dummy data if empty
            if (heartRateHistory.length === 0) {
                const initialHistory = Array.from({ length: 10 }, (_, i) => ({
                    time: new Date(Date.now() - (10 - i) * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
                    value: 60 + Math.random() * 20
                }));
                setHeartRateHistory(initialHistory);
            }

        } catch (error) {
            console.error('Load vitals error:', error);
            setVitals(null);
        } finally {
            setLoading(false);
        }
    };

    const renderVitalCard = (icon: string, title: string, value: string | number, unit: string, color: string, bgColor: string) => (
        <div className="bg-white dark:bg-card-dark p-4 rounded-xl shadow-sm border border-slate-100 dark:border-slate-800 flex items-center gap-4">
            <div className={`w-12 h-12 rounded-full flex items-center justify-center ${bgColor} ${color}`}>
                <span className="material-symbols-outlined">{icon}</span>
            </div>
            <div>
                <p className="text-sm text-slate-500 dark:text-slate-400">{title}</p>
                <div className="flex items-baseline gap-1">
                    <span className="text-2xl font-bold text-slate-900 dark:text-white">{value}</span>
                    <span className="text-sm text-slate-400">{unit}</span>
                </div>
            </div>
        </div>
    );

    return (
        <div className="min-h-screen bg-background-light dark:bg-background-dark pb-24 overflow-x-hidden">
            {/* Header */}
            <div className="bg-gradient-to-r from-slate-900 to-slate-800 p-6 pb-12 rounded-b-[2.5rem] shadow-xl">
                <div className="flex items-center justify-between mb-6">
                    <button
                        onClick={() => navigate(-1)}
                        className="p-2 bg-white/10 rounded-full text-white hover:bg-white/20 transition-colors"
                    >
                        <span className="material-symbols-outlined">arrow_back</span>
                    </button>
                    <h2 className="text-xl font-bold text-white">Smart Watch</h2>
                    <div className={`flex items-center gap-2 px-3 py-1 rounded-full ${isLive ? 'bg-green-500/20 text-green-400' : 'bg-slate-500/20 text-slate-400'}`}>
                        <div className={`w-2 h-2 rounded-full ${isLive ? 'bg-green-400 animate-pulse' : 'bg-slate-400'}`} />
                        <span className="text-xs font-medium">{isLive ? 'Live' : deviceStatus}</span>
                    </div>
                </div>

                {/* Device Card */}
                <div className="bg-white/10 backdrop-blur-md rounded-2xl p-6 border border-white/10 flex items-center justify-between">
                    <div>
                        <h3 className="text-white font-bold text-lg mb-1">
                            {bleConnected && smartwatchData.modelNumber
                                ? smartwatchData.modelNumber
                                : 'Smart Watch'}
                        </h3>
                        <p className="text-slate-300 text-sm">
                            {bleConnected && smartwatchData.batteryLevel != null
                                ? `Battery: ${smartwatchData.batteryLevel}%`
                                : bleConnected
                                    ? 'Connected via BLE'
                                    : 'Battery: --'}
                        </p>
                        {bleConnected && smartwatchData.manufacturerName && (
                            <p className="text-slate-400 text-xs mt-1">{smartwatchData.manufacturerName}</p>
                        )}
                    </div>
                    <span className="material-symbols-outlined text-4xl text-white/80">watch</span>
                </div>
            </div>

            <div className="px-4 -mt-8 space-y-6">
                {/* Vitals Grid */}
                {loading ? (
                    <div className="flex justify-center py-10">
                        <span className="w-8 h-8 border-4 border-primary border-t-transparent rounded-full animate-spin"></span>
                    </div>
                ) : vitals ? (
                    <div className="grid grid-cols-2 gap-3">
                        {renderVitalCard('monitor_heart', 'Heart Rate', vitals.heart_rate || '--', 'BPM', 'text-pink-500', 'bg-pink-50 dark:bg-pink-900/20')}
                        {renderVitalCard('bloodtype', 'Blood Pressure', vitals.blood_pressure || '--/--', 'mmHg', 'text-red-500', 'bg-red-50 dark:bg-red-900/20')}
                        {renderVitalCard('water_drop', 'SpO2', vitals.spo2 || '--', '%', 'text-cyan-500', 'bg-cyan-50 dark:bg-cyan-900/20')}
                        {renderVitalCard('thermostat', 'Temperature', vitals.temperature || '--', '°C', 'text-amber-500', 'bg-amber-50 dark:bg-amber-900/20')}
                        {renderVitalCard('directions_walk', 'Steps', vitals.steps?.toLocaleString() || '0', 'steps', 'text-blue-500', 'bg-blue-50 dark:bg-blue-900/20')}
                        {renderVitalCard('local_fire_department', 'Calories', vitals.calories || '0', 'kcal', 'text-orange-500', 'bg-orange-50 dark:bg-orange-900/20')}
                        {renderVitalCard('bedtime', 'Sleep', vitals.sleep || '--', '', 'text-purple-500', 'bg-purple-50 dark:bg-purple-900/20')}
                        {renderVitalCard('battery_full', 'Watch Battery', smartwatchData.batteryLevel ?? '--', '%', 'text-green-500', 'bg-green-50 dark:bg-green-900/20')}
                    </div>
                ) : (
                    <div className="text-center py-10 text-slate-500">No data available</div>
                )}

                {/* Live Chart */}
                <div className="bg-white dark:bg-card-dark rounded-2xl p-4 shadow-sm border border-slate-100 dark:border-slate-800">
                    <div className="flex items-center justify-between mb-4">
                        <h3 className="font-bold text-slate-800 dark:text-white flex items-center gap-2">
                            <span className="material-symbols-outlined text-pink-500">ecg_heart</span>
                            Live Heart Rate
                        </h3>
                        <span className="text-xs text-slate-400">Last 30 readings</span>
                    </div>

                    <div className="h-64 w-full min-w-0 min-h-0">
                        <ResponsiveContainer width="100%" height="100%" minWidth={0} minHeight={0} debounce={50}>
                            <AreaChart data={heartRateHistory}>
                                <defs>
                                    <linearGradient id="colorHr" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor="#ec4899" stopOpacity={0.3} />
                                        <stop offset="95%" stopColor="#ec4899" stopOpacity={0} />
                                    </linearGradient>
                                </defs>
                                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
                                <XAxis
                                    dataKey="time"
                                    hide={true}
                                />
                                <YAxis
                                    domain={['auto', 'auto']}
                                    orientation="right"
                                    tick={{ fontSize: 12, fill: '#94a3b8' }}
                                    axisLine={false}
                                    tickLine={false}
                                />
                                <Tooltip
                                    contentStyle={{
                                        backgroundColor: '#1e293b',
                                        border: 'none',
                                        borderRadius: '8px',
                                        color: '#fff'
                                    }}
                                    itemStyle={{ color: '#fff' }}
                                    labelStyle={{ display: 'none' }}
                                />
                                <Area
                                    type="monotone"
                                    dataKey="value"
                                    stroke="#ec4899"
                                    strokeWidth={3}
                                    fillOpacity={1}
                                    fill="url(#colorHr)"
                                    isAnimationActive={false}
                                />
                            </AreaChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                <button
                    onClick={loadVitals}
                    className="w-full py-4 bg-white dark:bg-card-dark rounded-xl font-bold text-slate-700 dark:text-white shadow-sm border border-slate-200 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-800 transition-colors"
                >
                    Sync Now
                </button>
            </div>
        </div>
    );
}
