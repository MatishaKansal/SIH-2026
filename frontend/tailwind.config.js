/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'ping-slow': 'ping 2s cubic-bezier(0, 0, 0.2, 1) infinite',
        'slide-down': 'slideDown 0.4s ease-out forwards',
        'glow-pulse': 'glowPulse 2s ease-in-out infinite',
        'spin-slow': 'spin 8s linear infinite',
        'counter-spin': 'spin 8s linear infinite reverse',
      },
      keyframes: {
        slideDown: {
          '0%': { transform: 'translateY(-20px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        glowPulse: {
          '0%, 100%': { boxShadow: '0 0 10px rgba(239,68,68,0.5)' },
          '50%': { boxShadow: '0 0 30px rgba(239,68,68,0.9), 0 0 60px rgba(239,68,68,0.4)' },
        },
      },
      boxShadow: {
        'glow-green': '0 0 20px rgba(52, 211, 153, 0.3)',
        'glow-red': '0 0 20px rgba(239, 68, 68, 0.4)',
        'glow-amber': '0 0 20px rgba(251, 191, 36, 0.3)',
        'glass': '0 8px 32px 0 rgba(0, 0, 0, 0.37)',
      },
      backdropBlur: {
        xs: '2px',
      },
      colors: {
        'slate-950': '#020617',
      }
    },
  },
  plugins: [],
}
