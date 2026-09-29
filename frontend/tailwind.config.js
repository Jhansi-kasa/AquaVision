/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
        mono: ['IBM Plex Mono', 'monospace'],
      },
      colors: {
        slate: {
          50: '#f8fafc',
          100: '#f1f5f9',
          200: '#e2e8f0',
          300: '#334155', // upgraded from faint #cbd5e1 to rich dark slate
          400: '#475569', // upgraded from washed out #94a3b8 to crisp slate-600
          500: '#334155', // upgraded from #64748b to sharp deep slate
          600: '#1e293b', // deep charcoal slate
          700: '#0f172a', // high-contrast slate-900
          800: '#091322', // obsidian navy
          900: '#020617',
        },
      },
      keyframes: {
        fadeUp: { '0%': { opacity: 0, transform: 'translateY(8px)' }, '100%': { opacity: 1, transform: 'translateY(0)' } },
        scan: { '0%': { transform: 'translateY(-100%)' }, '100%': { transform: 'translateY(100%)' } },
        popIn: { '0%': { opacity: 0, transform: 'scale(.94)' }, '100%': { opacity: 1, transform: 'scale(1)' } },
        slideIn: { '0%': { opacity: 0, transform: 'translateX(16px)' }, '100%': { opacity: 1, transform: 'translateX(0)' } },
      },
      animation: {
        fadeUp: 'fadeUp .45s cubic-bezier(.16,1,.3,1) both',
        scan: 'scan 2.4s linear infinite',
        popIn: 'popIn .25s cubic-bezier(.16,1,.3,1) both',
        slideIn: 'slideIn .25s cubic-bezier(.16,1,.3,1) both',
      },
    },
  },
  plugins: [],
};
