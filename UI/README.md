# LuddyBuddy Chat Widget

A simple React chat widget that doubles as a Chrome extension.

## Development

1. **Clone your repo** and enter the folder:
   ```bash
   git clone <your-repo-url>
   cd <app-folder>
   ```

2. **Install dependencies**:
   ```bash
   npm install        # or yarn install, or pnpm install
   ```
---

## Loading as a Chrome Extension

To install LuddyBuddy as a Chrome extension, follow these steps:

1. **Build the extension**
   ```bash
   pnpm run build      # or npm run build, yarn build
   ```

2. **Open Chrome’s Extensions page**
   - In your browser’s address bar, go to `chrome://extensions`.

3. **Enable Developer mode**
   - Toggle the **Developer mode** switch in the top-right corner of the page.

4. **Load the unpacked extension**
   - Click the **Load unpacked** button.
   - Select the root folder of your project (the one containing `manifest.json` and the `build` directory).

5. **Verify installation**
   - LuddyBuddy should now appear in your list of extensions and be ready to use.

