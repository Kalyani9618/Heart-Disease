import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
    View,
    Text,
    StyleSheet,
    TouchableOpacity,
    Switch,
    ScrollView,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import { apiClient } from '../services/apiClient';
import {
    requestNotificationPermission,
    scheduleLocalNotification,
    createNotificationChannels,
} from '../services/nativeNotificationService';
import { useToast } from '../components/Toast';

import { useAuth } from '../hooks/useAuth';

export default function NotificationScreen() {
    const navigate = useNavigate();
    const { user } = useAuth();
    const { showToast } = useToast();
    const [whatsappEnabled, setWhatsappEnabled] = useState(false);
    const [emailEnabled, setEmailEnabled] = useState(true);
    const [pushEnabled, setPushEnabled] = useState(true);
    const [loading, setLoading] = useState(false);

    const handleSave = async () => {
        if (!user) return;
        setLoading(true);
        try {
            // Request native notification permission when push is enabled
            if (pushEnabled) {
                const granted = await requestNotificationPermission();
                if (!granted) {
                    showToast('Notification permission denied. Please enable in device settings.', 'error');
                    setPushEnabled(false);
                    setLoading(false);
                    return;
                }
                // Ensure notification channels exist on Android
                await createNotificationChannels();
                // Register device token with backend
                await apiClient.registerDevice(user.id, 'native_push_token', 'android');
            }

            // Mock saving other preferences (email, whatsapp)
            await new Promise(resolve => setTimeout(resolve, 500));

            showToast('Notification preferences saved', 'success');
        } catch (error) {
            console.error('Save error:', error);
            showToast('Failed to save preferences', 'error');
        } finally {
            setLoading(false);
        }
    };

    const renderSettingItem = (
        icon: any,
        title: string,
        description: string,
        value: boolean,
        onValueChange: (val: boolean) => void,
        color: string
    ) => (
        <View style={styles.settingItem}>
            <View style={styles.settingInfo}>
                <View style={[styles.iconContainer, { backgroundColor: color }]}>
                    <Ionicons name={icon} size={24} color="#fff" />
                </View>
                <View style={styles.textContainer}>
                    <Text style={styles.settingTitle}>{title}</Text>
                    <Text style={styles.settingDescription}>{description}</Text>
                </View>
            </View>
            <Switch
                value={value}
                onValueChange={onValueChange}
                trackColor={{ false: '#e0e0e0', true: color }}
                thumbColor="#fff"
            />
        </View>
    );

    return (
        <SafeAreaView style={styles.container}>
            <LinearGradient colors={['#1a1a2e', '#16213e']} style={styles.header}>
                <TouchableOpacity onPress={() => navigate(-1)} style={styles.backButton}>
                    <Ionicons name="arrow-back" size={24} color="#fff" />
                </TouchableOpacity>
                <Text style={styles.headerTitle}>Notifications</Text>
                <View style={{ width: 24 }} />
            </LinearGradient>

            <ScrollView style={styles.content}>
                <View style={styles.section}>
                    <Text style={styles.sectionHeader}>Communication Channels</Text>

                    {renderSettingItem(
                        'logo-whatsapp',
                        'WhatsApp',
                        'Receive urgent alerts and daily summaries',
                        whatsappEnabled,
                        setWhatsappEnabled,
                        '#25D366'
                    )}

                    {renderSettingItem(
                        'mail',
                        'Email',
                        'Weekly reports and account updates',
                        emailEnabled,
                        setEmailEnabled,
                        '#EA4335'
                    )}

                    {renderSettingItem(
                        'notifications',
                        'Push Notifications',
                        'Real-time health alerts and reminders',
                        pushEnabled,
                        setPushEnabled,
                        '#4e54c8'
                    )}

                    {pushEnabled && (
                        <TouchableOpacity
                            style={{ marginTop: 10, backgroundColor: '#4e54c8', borderRadius: 8, padding: 10, alignItems: 'center' }}
                            onPress={async () => {
                                if (!user) return;
                                try {
                                    // Send a native local notification on the device
                                    await scheduleLocalNotification({
                                        id: Date.now(),
                                        title: 'Test Notification',
                                        body: 'This is a test push notification from Cardio AI.',
                                        scheduledAt: new Date(Date.now() + 1000), // 1 second from now
                                        channelId: 'cardio-reminders',
                                    });
                                    showToast('Test notification sent!', 'success');
                                } catch (e) {
                                    // Fallback to backend push if native fails
                                    try {
                                        await apiClient.sendPushNotification({
                                            user_id: user.id,
                                            title: 'Test Notification',
                                            body: 'This is a test push notification from Cardio AI.'
                                        });
                                        showToast('Test notification sent via server!', 'success');
                                    } catch (err) {
                                        showToast('Failed to send test notification', 'error');
                                    }
                                }
                            }}
                        >
                            <Text style={{ color: '#fff', fontWeight: 'bold' }}>Send Test Notification</Text>
                        </TouchableOpacity>
                    )}
                </View>

                <View style={styles.infoCard}>
                    <Ionicons name="information-circle" size={24} color="#4e54c8" />
                    <Text style={styles.infoText}>
                        Urgent health alerts will always be sent via Push Notification regardless of your preferences.
                    </Text>
                </View>

                <TouchableOpacity
                    style={styles.saveButton}
                    onPress={handleSave}
                    disabled={loading}
                >
                    <LinearGradient
                        colors={['#4e54c8', '#8f94fb']}
                        style={styles.gradientButton}
                    >
                        <Text style={styles.buttonText}>
                            {loading ? 'Saving...' : 'Save Preferences'}
                        </Text>
                    </LinearGradient>
                </TouchableOpacity>
            </ScrollView>
        </SafeAreaView>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: '#f5f5f5',
    },
    header: {
        padding: 20,
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
    },
    backButton: {
        padding: 5,
    },
    headerTitle: {
        fontSize: 20,
        fontWeight: 'bold',
        color: '#fff',
    },
    content: {
        flex: 1,
        padding: 20,
    },
    section: {
        backgroundColor: '#fff',
        borderRadius: 15,
        padding: 20,
        marginBottom: 20,
        boxShadow: '0 1px 2px rgba(0, 0, 0, 0.1)',
    },
    sectionHeader: {
        fontSize: 18,
        fontWeight: 'bold',
        color: '#333',
        marginBottom: 20,
    },
    settingItem: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        marginBottom: 25,
    },
    settingInfo: {
        flexDirection: 'row',
        alignItems: 'center',
        flex: 1,
        marginRight: 10,
    },
    iconContainer: {
        width: 40,
        height: 40,
        borderRadius: 20,
        alignItems: 'center',
        justifyContent: 'center',
        marginRight: 15,
    },
    textContainer: {
        flex: 1,
    },
    settingTitle: {
        fontSize: 16,
        fontWeight: '600',
        color: '#333',
        marginBottom: 4,
    },
    settingDescription: {
        fontSize: 12,
        color: '#888',
    },
    infoCard: {
        flexDirection: 'row',
        backgroundColor: '#e8eaf6',
        padding: 15,
        borderRadius: 12,
        alignItems: 'center',
        marginBottom: 30,
    },
    infoText: {
        flex: 1,
        marginLeft: 10,
        color: '#333',
        fontSize: 14,
        lineHeight: 20,
    },
    saveButton: {
        borderRadius: 15,
        overflow: 'hidden',
        boxShadow: '0 2px 4px rgba(0, 0, 0, 0.25)',
    },
    gradientButton: {
        padding: 18,
        alignItems: 'center',
    },
    buttonText: {
        color: '#fff',
        fontSize: 18,
        fontWeight: 'bold',
    },
});
