# Portfolio Daily Report Automation

Automated daily portfolio tracker that sends you an email at market close with:
- Portfolio value and daily return
- Performance vs benchmarks (S&P 500 TR, S&P 500 EW, MSCI World, MSCI World EW)
- Strategy attribution (which strategies contributed most to gains/losses)
- Top winners and losers
- Full breakdown by strategy

## Quick Start

### 1. Test Locally First

```bash
cd portfolio-tracker
pip install -r requirements.txt
python test_local.py
```

This will fetch live data and open a sample report in your browser.

### 2. Set Up Gmail App Password

1. Go to https://myaccount.google.com/security
2. Enable 2-Step Verification (if not already enabled)
3. Go to https://myaccount.google.com/apppasswords
4. Select "Mail" and "Windows Computer"
5. Click "Generate"
6. Copy the 16-character password (e.g., `abcd efgh ijkl mnop`)

### 3. Test Email Locally

```bash
# Set environment variables (PowerShell)
$env:GMAIL_SENDER = "your.email@gmail.com"
$env:GMAIL_RECIPIENT = "your.email@gmail.com"
$env:GMAIL_APP_PASSWORD = "your-16-char-app-password"

# Run the report
python daily_report.py
```

### 4. Deploy to GitHub

1. Create a new GitHub repository (can be private)
2. Push this folder to the repo:
   ```bash
   git init
   git add .
   git commit -m "Initial portfolio tracker"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/portfolio-tracker.git
   git push -u origin main
   ```

3. Add secrets to your GitHub repo:
   - Go to repo Settings > Secrets and variables > Actions
   - Add these secrets:
     - `GMAIL_SENDER`: your Gmail address
     - `GMAIL_RECIPIENT`: email to receive reports (can be same)
     - `GMAIL_APP_PASSWORD`: the 16-character app password

4. The workflow will run automatically at 4:05 PM ET on weekdays

### 5. Manual Test on GitHub

Go to Actions > "Daily Portfolio Report" > "Run workflow" to test immediately.

## Configuration

Edit `portfolio_config.json` to update your holdings:

```json
{
  "strategies": {
    "Strategy Name": {
      "description": "Why this strategy",
      "holdings": [
        {"ticker": "AAPL", "shares": 100}
      ]
    }
  }
}
```

## Files

- `portfolio_config.json` - Your holdings and strategy mappings
- `daily_report.py` - Main script that generates and sends reports
- `test_local.py` - Test script to preview reports locally
- `.github/workflows/daily_report.yml` - GitHub Actions schedule

## Troubleshooting

**"No price data for ticker"**
- Check the ticker symbol is correct for yfinance
- European stocks need exchange suffix (e.g., `BESI.AS` for Amsterdam)
- Hong Kong stocks use `.HK` suffix

**Email not sending**
- Verify app password is correct (16 chars, no spaces)
- Check GitHub Actions logs for errors
- Ensure 2FA is enabled on Gmail

**Wrong schedule time**
- Cron uses UTC. 21:05 UTC = 4:05 PM ET (during EST, adjust for DST)
