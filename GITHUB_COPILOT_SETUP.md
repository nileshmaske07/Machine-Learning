# GitHub Copilot Setup and Troubleshooting Guide

This guide helps you set up GitHub Copilot for developing Machine Learning projects and resolve common subscription and VSCode integration issues.

## 🔍 Check Your Current Copilot Subscription

### Method 1: Via GitHub Web Interface
1. Go to [GitHub.com](https://github.com)
2. Click your profile picture → **Settings**
3. In the left sidebar, click **Copilot**
4. View your current plan status and billing information

### Method 2: Via VSCode
1. Open VSCode
2. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
3. Type "GitHub Copilot: Check Status"
4. Run the command to see your subscription status

### Method 3: Via GitHub CLI
```bash
gh copilot status
```

## 💳 GitHub Copilot Subscription Plans

### Available Plans:
- **GitHub Copilot Individual**: $10/month or $100/year
- **GitHub Copilot Business**: $19/month per user
- **GitHub Copilot Enterprise**: $39/month per user

### How to Subscribe:

#### For Individual Plan:
1. Visit [GitHub Copilot pricing page](https://github.com/features/copilot#pricing)
2. Click "Get GitHub Copilot"
3. Choose "For individuals" 
4. Select monthly or yearly billing
5. Complete payment setup

#### Alternative Method:
1. Go to GitHub Settings → Copilot
2. Click "Enable GitHub Copilot"
3. Choose your plan and payment method

## 🔧 VSCode Setup for GitHub Copilot

### Required Extensions:
1. **GitHub Copilot** - Main extension
2. **GitHub Copilot Chat** - For interactive AI assistance

### Installation Steps:
1. Open VSCode
2. Go to Extensions (Ctrl+Shift+X)
3. Search for "GitHub Copilot"
4. Install both extensions
5. Restart VSCode
6. Sign in to GitHub when prompted

## ❌ Common Errors and Solutions

### Error: "GitHub Copilot is not available"
**Solutions:**
- Ensure you have an active subscription
- Check your internet connection
- Sign out and sign back into GitHub in VSCode
- Restart VSCode

### Error: "Authentication failed"
**Solutions:**
```bash
# Clear GitHub credentials
gh auth logout
gh auth login
```
Or in VSCode:
- Command Palette → "GitHub: Sign out"
- Command Palette → "GitHub: Sign in"

### Error: "Copilot suggestions not appearing"
**Solutions:**
1. Check if Copilot is enabled:
   - Command Palette → "GitHub Copilot: Toggle"
2. Verify file type support (works with most programming languages)
3. Check VSCode settings:
   ```json
   {
     "github.copilot.enable": {
       "*": true,
       "yaml": false,
       "plaintext": false,
       "markdown": true,
       "python": true
     }
   }
   ```

### Error: "Rate limit exceeded"
**Solutions:**
- Wait for the rate limit to reset (usually 1 hour)
- Upgrade to a higher tier plan if needed
- Check your subscription status

## 🐍 Using Copilot with Machine Learning Projects

### Recommended VSCode Settings for ML Development:
```json
{
  "python.defaultInterpreterPath": "./venv/bin/python",
  "github.copilot.enable": {
    "python": true,
    "jupyter": true,
    "markdown": true
  },
  "jupyter.askForKernelRestart": false,
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true
}
```

### Useful Copilot Commands for ML:
- `Ctrl+I` - Inline suggestions
- `Alt+]` - Next suggestion
- `Alt+[` - Previous suggestion
- `Tab` - Accept suggestion
- `Esc` - Dismiss suggestion

### Example Copilot Usage in this Repository:
```python
# Start typing a comment and let Copilot suggest implementations
# Load and preprocess data for decision tree
# Train decision tree classifier with optimal parameters
# Evaluate model performance using cross-validation
```

## 🆘 Still Having Issues?

### Check These Common Causes:
1. **Network restrictions** - Corporate firewalls may block Copilot
2. **Proxy settings** - Configure VSCode proxy if needed
3. **Outdated extensions** - Update GitHub Copilot extensions
4. **Account permissions** - Ensure your GitHub account has proper access

### VSCode Diagnostic Commands:
```
Developer: Toggle Developer Tools (to check console errors)
GitHub Copilot: Check Status
GitHub Copilot: Show Output
```

### Contact Support:
- GitHub Support: [support.github.com](https://support.github.com)
- VSCode Issues: [GitHub VSCode Repository](https://github.com/microsoft/vscode/issues)
- Copilot Issues: [GitHub Copilot Discussions](https://github.com/orgs/community/discussions/categories/copilot)

## 📝 Quick Setup Checklist

- [ ] Active GitHub Copilot subscription
- [ ] VSCode GitHub Copilot extension installed
- [ ] VSCode GitHub Copilot Chat extension installed
- [ ] Signed into GitHub account in VSCode
- [ ] Python extension installed for ML development
- [ ] Jupyter extension installed for notebook support
- [ ] Test Copilot suggestions working in Python files

---

**Note**: This repository contains Machine Learning implementations in Python. GitHub Copilot can significantly speed up your ML development by suggesting code completions, documentation, and test cases.