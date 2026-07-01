import os
import logging
import io
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Configuration
BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
if not BOT_TOKEN:
    logger.error("TELEGRAM_BOT_TOKEN not set!")
    exit(1)

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')

# Image generation function
async def generate_image(prompt):
    """Generate image using available AI provider"""
    
    # Try OpenAI first
    if OPENAI_API_KEY:
        try:
            import openai
            import requests
            
            openai.api_key = OPENAI_API_KEY
            logger.info(f"Generating image with OpenAI: {prompt[:50]}...")
            
            response = openai.images.generate(
                model="dall-e-2",
                prompt=prompt,
                n=1,
                size="512x512"
            )
            
            image_url = response.data[0].url
            img_response = requests.get(image_url, timeout=30)
            logger.info("Image generated successfully with OpenAI")
            return img_response.content
            
        except Exception as e:
            logger.error(f"OpenAI error: {e}")
            # Fall through to next provider
    
    # Try Gemini if OpenAI fails or not configured
    if GEMINI_API_KEY:
        try:
            import google.generativeai as genai
            
            genai.configure(api_key=GEMINI_API_KEY)
            logger.info(f"Generating image with Gemini: {prompt[:50]}...")
            
            model = genai.GenerativeModel('gemini-2.0-flash-exp-image-generation')
            response = model.generate_content(f"Generate an image of: {prompt}")
            
            if response and response.candidates:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'inline_data') and part.inline_data.data:
                        logger.info("Image generated successfully with Gemini")
                        return part.inline_data.data
            
            logger.error("No image data in Gemini response")
            return None
            
        except Exception as e:
            logger.error(f"Gemini error: {e}")
            return None
    
    logger.error("No AI provider available!")
    return None

# Start command
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command"""
    user = update.effective_user
    await update.message.reply_text(
        f"🎨 **Welcome to ArtForgeBot, {user.first_name}!**\n\n"
        "I create images from your text descriptions using AI.\n\n"
        "**How to use:**\n"
        "Just send me any text description, and I'll generate an image!\n\n"
        "**Example prompts:**\n"
        "• 'A beautiful sunset over mountains'\n"
        "• 'A cute cat wearing a wizard hat'\n"
        "• 'Cyberpunk city with neon lights'\n\n"
        "**Commands:**\n"
        "/start - Show this message\n"
        "/help - Get help\n"
        "/status - Check bot status",
        parse_mode='Markdown'
    )

# Help command
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /help command"""
    await update.message.reply_text(
        "🖼️ **ArtForgeBot Help**\n\n"
        "**How to generate images:**\n"
        "1. Type any description\n"
        "2. Wait 10-20 seconds\n"
        "3. Receive your image!\n\n"
        "**Tips:**\n"
        "• Be descriptive\n"
        "• Include style (photorealistic, anime, etc.)\n"
        "• Mention colors and mood\n\n"
        "**Commands:**\n"
        "/start - Welcome\n"
        "/help - This help\n"
        "/status - Check status",
        parse_mode='Markdown'
    )

# Status command
async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /status command"""
    providers = []
    if OPENAI_API_KEY:
        providers.append("✅ OpenAI")
    if GEMINI_API_KEY:
        providers.append("✅ Google Gemini")
    
    if not providers:
        providers.append("❌ No AI providers configured")
    
    status_text = (
        "🔍 **Bot Status**\n\n"
        f"🤖 Username: @ArtForgeBot\n"
        f"🟢 Status: Online\n"
        f"📊 Providers: {', '.join(providers)}\n"
        f"📦 Version: 2.0"
    )
    
    await update.message.reply_text(status_text, parse_mode='Markdown')

# Handle text messages
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle regular text messages"""
    prompt = update.message.text
    
    # Ignore very short messages
    if len(prompt) < 3:
        await update.message.reply_text("❌ Please provide a longer description.")
        return
    
    # Send processing message
    processing_msg = await update.message.reply_text(
        f"🎨 **Generating image...**\n\n"
        f"📝 Prompt: _{prompt[:100]}_\n\n"
        "⏳ Please wait 10-20 seconds..."
    )
    
    try:
        # Generate image
        image_data = await generate_image(prompt)
        
        if image_data:
            await processing_msg.delete()
            await update.message.reply_photo(
                photo=io.BytesIO(image_data),
                caption=f"🖼️ **Generated Image**\n\n📝 {prompt[:200]}\n\n🤖 @ArtForgeBot",
                parse_mode='Markdown'
            )
        else:
            await processing_msg.edit_text(
                "❌ **Generation Failed**\n\n"
                "Couldn't generate image. Please try:\n"
                "• Using a different description\n"
                "• Making it more specific\n"
                "• Trying again later"
            )
            
    except Exception as e:
        logger.error(f"Error: {e}")
        await processing_msg.edit_text(
            "❌ **Error Occurred**\n\n"
            "Something went wrong. Please try again."
        )

# Error handler
async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle errors"""
    logger.error(f"Update {update} caused error {context.error}")
    if update and update.effective_message:
        await update.effective_message.reply_text(
            "❌ An error occurred. Please try again later."
        )

# Main function
def main():
    """Start the bot"""
    logger.info("=" * 50)
    logger.info("🚀 Starting ArtForgeBot...")
    logger.info(f"🤖 Username: @ArtForgeBot")
    logger.info(f"📊 OpenAI: {'✅' if OPENAI_API_KEY else '❌'}")
    logger.info(f"📊 Gemini: {'✅' if GEMINI_API_KEY else '❌'}")
    
    if not OPENAI_API_KEY and not GEMINI_API_KEY:
        logger.error("❌ No AI providers configured! Exiting...")
        return
    
    # Create application
    application = Application.builder().token(BOT_TOKEN).build()
    
    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("status", status))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_error_handler(error_handler)
    
    # Start the bot
    logger.info("✅ Bot is running! Press Ctrl+C to stop.")
    logger.info("=" * 50)
    application.run_polling()

if __name__ == "__main__":
    main()
