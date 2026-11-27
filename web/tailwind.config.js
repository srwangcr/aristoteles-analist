/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'SFMono-Regular', 'monospace'],
      },
      colors: {
        obsidian: {
          950: '#05070d',
          900: '#0b101b',
          850: '#101827',
          800: '#162033',
          700: '#223049',
        },
      },
      boxShadow: {
        panel: '0 18px 40px -24px rgba(0, 0, 0, 0.9)',
      },
    },
  },
  plugins: [],
};
