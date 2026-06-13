import path from 'path';
import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
import reactNativeWeb from 'vite-plugin-react-native-web';


export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, '.', '');
  const isProd = mode === 'production';

  return {
    // Use relative base path for Capacitor mobile compatibility
    base: './',
    server: {
      port: parseInt(env.VITE_FRONTEND_PORT) || 3000,
      host: '0.0.0.0',
      strictPort: true,
      proxy: {
        '/api': {
          target: 'http://localhost:5001',
          changeOrigin: true,
          secure: false,
          timeout: 300000, // 5 min - allow long-running research/search requests
          proxyTimeout: 300000,
        },
      },
    },
    plugins: [
      react({
        jsxRuntime: 'automatic',
        jsxImportSource: 'react',
      }),
      reactNativeWeb()
    ],
    esbuild: {
      // Force production JSX transform
      jsx: 'automatic',
      jsxDev: false,
    },
    define: {
      // SECURITY: API keys must NOT be inlined in the client bundle.
      // All AI API calls should go through the backend which holds keys server-side.
      global: 'globalThis',
      __DEV__: mode === 'development',
    },
    resolve: {
      alias: {
        '@': path.resolve(__dirname, '.'),
        'react-native': 'react-native-web',
        'react-native-linear-gradient': 'expo-linear-gradient',
        'react-native-safe-area-context': path.resolve(__dirname, 'components/SafeAreaView.tsx'),
        'react-native/Libraries/Utilities/codegenNativeComponent': 'react-native-web/dist/exports/View',
        '@react-native/assets-registry/registry': path.resolve(__dirname, 'mocks/assets-registry.ts'),
        'invariant': path.resolve(__dirname, 'mocks/invariant.ts'),
        'fontfaceobserver': path.resolve(__dirname, 'mocks/fontfaceobserver.ts'),
        'expo-modules-core': path.resolve(__dirname, 'mocks/expo-modules-core.ts'),
        'expo-font': path.resolve(__dirname, 'mocks/expo-font.ts'),
        'expo-image-picker': path.resolve(__dirname, 'mocks/expo-image-picker.ts'),
        'expo-image-manipulator': path.resolve(__dirname, 'mocks/expo-image-manipulator.ts'),
        '@capacitor/filesystem': path.resolve(__dirname, 'mocks/capacitor-filesystem.ts'),
      },
      extensions: ['.web.js', '.web.jsx', '.web.ts', '.web.tsx', '.js', '.jsx', '.ts', '.tsx', '.json'],
    },
    optimizeDeps: {
      esbuildOptions: {
        resolveExtensions: ['.web.js', '.web.jsx', '.web.ts', '.web.tsx', '.js', '.jsx', '.ts', '.tsx', '.json'],
        loader: {
          '.js': 'jsx',
          '.ts': 'tsx',
        },
        define: {
          global: 'globalThis',
        },
      },
      include: [
        'react',
        'react-dom',
        'react-native-web',
        '@expo/vector-icons',
      ],
      exclude: [
        '@react-native/assets-registry',
        'expo',
        'expo-modules-core',
        'expo-linear-gradient',
        'expo-image-manipulator',
        '@capacitor/filesystem',
      ],
    },
    build: {
      rollupOptions: {
        external: ['expo-image-manipulator'],
        output: {
          manualChunks: {
            'vendor-react': ['react', 'react-dom', 'react-router-dom'],
            'vendor-rn': ['react-native-web', '@expo/vector-icons'],
            'data-misc': ['./data/gamification.ts', './data/translations.ts'],
          },
        },
      },
      // CSS code splitting for optimal loading
      cssCodeSplit: true,
      minify: 'terser',
      terserOptions: {
        compress: {
          drop_console: false,
        },
      },
    },
  };
});
