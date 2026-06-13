/**
 * SafeAreaView Shim for Vite Web Build
 * 
 * Provides a web-compatible replacement for react-native-safe-area-context.
 * Used as a Vite alias target in vite.config.ts.
 */
import React from 'react';
import { View } from 'react-native';

export const SafeAreaView = ({ children, style, ...props }: any) => (
  <View style={style} {...props}>{children}</View>
);

export const SafeAreaProvider = ({ children }: any) => <>{children}</>;

export const useSafeAreaInsets = () => ({ top: 0, bottom: 0, left: 0, right: 0 });

export default SafeAreaView;
