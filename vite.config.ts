import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    // getUserMedia and WebGL float rendering both want a secure context. localhost counts as one,
    // but phones on the LAN do not — run `vite --host` behind a tunnel if you need a device test.
    host: true,
  },
  assetsInclude: ['**/*.dnw'],
});
