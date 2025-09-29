# Telegram Generative AI Coding Agent (Vercel Edition)

This is a resilient, generative AI assistant built on the modern `aiogram` framework. It is designed for simple, robust deployment as a serverless function on the Vercel platform, using a persistent PostgreSQL database from Neon for memory.

## Core Features

- **Serverless Architecture**: Runs as a proper ASGI application on Vercel, ensuring scalability and efficiency.
- **Persistent Memory**: Uses a PostgreSQL database (e.g., from Neon) to remember conversation history and execution results, making it stateful across serverless invocations.
- **Resilient AI**: Uses Google's Gemini model as the primary AI provider and automatically fails over to Groq's Llama3 if the primary model is unavailable.
- **Code Execution**: Directly execute Python, JavaScript, and Bash code within a secure environment.
- **AI-Powered Debugging**: If code execution fails, the bot can use its AI to analyze the error and suggest a fix.

## Deployment Instructions

Follow these steps to deploy and run your own instance of the bot.

### 1. Set Up Environment Variables on Vercel

Before deploying, you need to configure the necessary secrets in your Vercel project.

1.  Go to your project's dashboard on Vercel.
2.  Navigate to **Settings** > **Environment Variables**.
3.  Add the following variables:

| Variable Name    | Description                                                              |
| ---------------- | ------------------------------------------------------------------------ |
| `BOT_TOKEN`      | Your Telegram Bot API token, obtained from [@BotFather](https://t.me/BotFather). |
| `DATABASE_URL`   | The connection string for your PostgreSQL database (e.g., from Neon).    |
| `VERCEL_URL`     | The public domain of your Vercel deployment (e.g., `your-app.vercel.app`). |
| `GEMINI_API_KEY` | Your API key for the Google Gemini AI model.                             |
| `GROQ_API_KEY`   | Your API key for the Groq AI model (used as a fallback).                 |
| `WEBHOOK_SECRET` | A random, secure string you create to verify webhook requests.           |

### 2. Deploy to Vercel

Once your environment variables are set, deploy your project. You can do this by connecting your Git repository (e.g., GitHub, GitLab) to Vercel. Vercel will automatically detect the `vercel.json` configuration and build the application.

### 3. Set the Telegram Webhook

After the deployment is live, you need to tell Telegram where to send updates. This is done by running the `setup_bot.py` script from your local machine.

1.  **Clone the repository** to your local machine if you haven't already.
2.  **Create a `.env` file** in the root of the project with the same `BOT_TOKEN` and `VERCEL_URL` you set in Vercel. This is only for running the setup script locally.
    ```
    BOT_TOKEN="your-telegram-bot-token"
    VERCEL_URL="your-app.vercel.app"
    ```
3.  **Install dependencies** locally:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the setup script**:
    ```bash
    python setup_bot.py
    ```

The script will register your Vercel deployment URL with Telegram. Your bot is now live and ready to receive messages!