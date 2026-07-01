#!/usr/bin/env python3
"""
ArtForgeBot - AI Image Generation Telegram Bot
Deployed on Railway with GitHub integration
"""

import os
import sys
import logging
import io
import asyncio
from typing import Optional, Dict, Any
from datetime import datetime
import traceback

# Third-party imports
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters
)
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================
# LOGGING CONFIGURATION
# ============================================

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ============================================
# CONFIGURATION
# ============================================

class Config:
    """Bot configuration from environment variables"""
    
    # Required: Telegram Bot Token
    BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
    if not BOT_TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN environment variable is required!")
    
    # Optional: AI Provider API Keys
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
    STABILITY_API_KEY = os.environ.get('STABILITY_API_KEY')
    
    # Optional: Bot settings
    MAX_PROMPT_LENGTH = 500
    GENERATION_TIMEOUT = 60  # seconds
    MAX_RETRIES = 3
    
    @classmethod
    def get_active_providers(cls) -> list:
        """Get list of configured AI providers"""
        providers = []
        if cls.OPENAI_API_KEY:
            providers.append("OpenAI DALL-E")
        if cls.GEMINI_API_KEY:
            providers.append("Google Gemini")
        if cls.STABILITY_API_KEY:
            providers.append("Stability AI")
        return providers
    
    @classmethod
    def is_configured(cls) -> bool:
        """Check if at least one AI provider is configured"""
        return bool(cls.get_active_providers())

# ============================================
# AI PROVIDER HANDLERS
# ============================================

class ImageGenerator:
    """Handle image generation from different AI providers"""
    
    @staticmethod
    async def generate_with_openai(prompt: str) -> Optional[bytes]:
        """Generate image using OpenAI DALL-E models"""
        try:
            import openai
            import requests
            
            openai.api_key = Config.OPENAI_API_KEY
            
            # Try DALL-E 3 first, fallback to DALL-E 2
            models = ["dall-e-3", "dall-e-2"]
            
            for model in models:
                try:
                    logger.info(f"Attempting generation with {model}")
                    
                    # Prepare parameters based on model
                    params = {
                        "model": model,
                        "prompt": prompt,
                        "n": 1,
                        "size": "1024x1024"
                    }
                    
                    # DALL-E 3 specific parameters
                    if model == "dall-e-3":
                        params["quality"] = "standard"
                    
                    response = openai.images.generate(**params)
                    image_url = response.data[0].url
                    
                    # Download the image
                    img_response = requests.get(image_url, timeout=30)
                    if img_response.status_code == 200:
                        logger.info(f"Successfully generated image with {model}")
                        return img_response.content
                        
                except Exception as e:
                    logger.warning(f"Failed with {model}: {str(e)}")
                    continue
            
            logger.error("All OpenAI models failed")
            return None
            
        except Exception as e:
            logger.error(f"OpenAI generation error: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    @staticmethod
    async def generate_with_gemini(prompt: str) -> Optional[bytes]:
        """Generate image using Google Gemini"""
        try:
            import google.generativeai as genai
            
            genai.configure(api_key=Config.GEMINI_API_KEY)
            
            # Use Gemini 2.0 Flash with image generation capability
            model = genai.GenerativeModel('gemini-2.0-flash-exp-image-generation')
            
            response = model.generate_content(
                f"Generate a detailed, high-quality image of: {prompt}",
                generation_config={
                    "temperature": 1.0,
                    "candidate_count": 1,
                    "max_output_tokens": 2048
                }
            )
            
            # Extract image data from response
            if response and response.candidates:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'inline_data') and part.inline_data.data:
                        logger.info("Successfully generated image with Gemini")
                        return part.inline_data.data
            
            logger.error("No image data found in Gemini response")
            return None
            
        except Exception as e:
            logger.error(f"Gemini generation error: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    @staticmethod
    async def generate_with_stability(prompt: str) -> Optional[bytes]:
        """Generate image using Stability AI"""
        try:
            import requests
            import base64
            
            headers = {
                "Authorization": f"Bearer {Config.STABILITY_API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "text_prompts": [{"text": prompt}],
                "cfg_scale": 7,
                "clip_guidance_preset": "FAST_BLUE",
                "height": 1024,
                "width": 1024,
                "samples": 1,
                "steps": 30
            }
            
            response = requests.post(
                "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('artifacts'):
                    image_data = base64.b64decode(data['artifacts'][0]['base64'])
                    logger.info("Successfully generated image with Stability AI")
                    return image_data
            else:
                logger.error(f"Stability API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Stability AI generation error: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    @classmethod
    async def generate(cls, prompt: str) -> Optional[bytes]:
        """
        Generate image using the first available AI provider
        Priority: OpenAI > Gemini > Stability AI
        """
        if not Config.is_configured():
            logger.error("No AI provider configured!")
            return None
        
        # Try providers in priority order
        if Config.OPENAI_API_KEY:
            result = await cls.generate_with_openai(prompt)
            if result:
                return result
        
        if Config.GEMINI_API_KEY:
            result = await cls.generate_with_gemini(prompt)
            if result:
                return result
        
        if Config.STABILITY_API_KEY:
            result = await cls.generate_with_stability(prompt)
            if result:
                return result
        
        logger.error("All AI providers failed to generate image")
        return None

# ============================================
# TELEGRAM BOT HANDLERS
# ============================================

class BotHandlers:
    """Telegram bot command handlers"""
    
    @staticmethod
    async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start command"""
        user = update.effective_user
        username = user.username or user.first_name
        
        # Create inline keyboard
        keyboard = [
            [
                InlineKeyboardButton("🎨 Generate", callback_data="generate"),
                InlineKeyboardButton("ℹ️ Help", callback_data="help")
            ],
            [
                InlineKeyboardButton("📊 Status", callback_data="status"),
                InlineKeyboardButton("📝 Examples", callback_data="examples")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        welcome_message = (
            f"🎨 **Welcome to ArtForgeBot, {username}!**\n\n"
            "I create stunning images from your text descriptions using advanced AI.\n\n"
            "**✨ Quick Start:**\n"
            "Simply send me any text description and I'll generate an image!\n\n"
            "**📌 Commands:**\n"
            "/start - Show this menu\n"
            "/help - Detailed help\n"
            "/status - Check bot status\n"
            "/generate <prompt> - Generate an image\n"
            "/examples - See example prompts\n\n"
            "⚡ **Powered by:** " + ", ".join(Config.get_active_providers())
        )
        
        await update.message.reply_text(
            welcome_message,
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
    
    @staticmethod
    async def help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /help command"""
        help_text = (
            "🖼️ **ArtForgeBot - Help Center**\n\n"
            "**🎯 How to generate images:**\n"
            "1. Type a description of what you want\n"
            "2. Wait 10-30 seconds for generation\n"
            "3. Receive your AI-generated image!\n\n"
            "**💡 Tips for best results:**\n"
            "• Be specific and descriptive\n"
            "• Include art style (photorealistic, anime, painting)\n"
            "• Mention colors, lighting, and mood\n"
            "• Add composition details\n\n"
            "**🎨 Style examples:**\n"
            "• Photorealistic\n"
            "• Anime/Manga\n"
            "• Watercolor painting\n"
            "• Cyberpunk\n"
            "• Fantasy art\n"
            "• Minimalist\n\n"
            "**📌 Commands:**\n"
            "/start - Welcome message\n"
            "/help - This help menu\n"
            "/status - Check bot status\n"
            "/generate <prompt> - Generate image\n"
            "/examples - See example prompts\n\n"
            "**🔒 Privacy:** Your prompts are not stored permanently."
        )
        
        await update.message.reply_text(help_text, parse_mode='Markdown')
    
    @staticmethod
    async def status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /status command"""
        providers = Config.get_active_providers()
        
        status_text = (
            "🔍 **ArtForgeBot Status**\n\n"
            f"🤖 Bot: @ArtForgeBot\n"
            f"🟢 Status: Online\n"
            f"📊 Providers: {', '.join(providers) if providers else 'None'}\n"
            f"⏰ Uptime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )
        
        if providers:
            status_text += "✅ **Configuration:** All systems operational!"
        else:
            status_text += "❌ **Warning:** No AI providers configured!\n"
            status_text += "Please contact the administrator."
        
        await update.message.reply_text(status_text, parse_mode='Markdown')
    
    @staticmethod
    async def examples(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /examples command"""
        examples = (
            "📝 **Example Prompts to Try**\n\n"
            "**Fantasy:**\n"
            "• 'A majestic dragon flying over snow-capped mountains at sunset'\n"
            "• 'An enchanted forest with glowing mushrooms and fairy lights'\n\n"
            "**Sci-Fi:**\n"
            "• 'Cyberpunk cityscape with neon lights and flying cars'\n"
            "• 'A futuristic space station orbiting a purple gas giant'\n\n"
            "**Animals:**\n"
            "• 'A cute orange cat wearing a wizard hat and cape'\n"
            "• 'A majestic wolf howling at the full moon'\n\n"
            "**Portraits:**\n"
            "• 'Photorealistic portrait of a wise old wizard'\n"
            "• 'Anime girl with pink hair in a cyberpunk outfit'\n\n"
            "**Landscapes:**\n"
            "• 'A serene lake reflecting the northern lights'\n"
            "• 'A vibrant autumn forest with golden leaves'\n\n"
            "💡 **Tip:** Try combining styles and details for unique results!"
        )
        
        await update.message.reply_text(examples, parse_mode='Markdown')
    
    @staticmethod
    async def generate(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /generate command"""
        # Extract prompt from command
        prompt = " ".join(context.args) if context.args else ""
        
        if not prompt:
            await update.message.reply_text(
                "❌ **Please provide a prompt!**\n\n"
                "Usage: /generate <description>\n\n"
                "Example: /generate A beautiful sunset over the ocean"
            )
            return
        
        await BotHandlers._process_generation(update, prompt)
    
    @staticmethod
    async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle regular text messages"""
        prompt = update.message.text
        
        # Ignore very short messages
        if len(prompt) < 3:
            await update.message.reply_text(
                "❌ Please provide a longer description (at least 3 characters)."
            )
            return
        
        # Limit prompt length
        if len(prompt) > Config.MAX_PROMPT_LENGTH:
            prompt = prompt[:Config.MAX_PROMPT_LENGTH]
            await update.message.reply_text(
                f"⚠️ Prompt truncated to {Config.MAX_PROMPT_LENGTH} characters."
            )
        
        await BotHandlers._process_generation(update, prompt)
    
    @staticmethod
    async def _process_generation(update: Update, prompt: str) -> None:
        """Process image generation request"""
        # Check if AI is configured
        if not Config.is_configured():
            await update.message.reply_text(
                "❌ **Bot Not Configured**\n\n"
                "No AI providers are configured. Please contact the administrator."
            )
            return
        
        # Send initial processing message
        processing_msg = await update.message.reply_text(
            f"🎨 **Generating your image...**\n\n"
            f"📝 Prompt: _{prompt[:150]}_\n\n"
            "⏳ This may take 10-30 seconds...\n"
            "Please wait, I'm creating your artwork! 🎨"
        )
        
        try:
            # Generate the image with timeout
            image_data = await asyncio.wait_for(
                ImageGenerator.generate(prompt),
                timeout=Config.GENERATION_TIMEOUT
            )
            
            if image_data:
                # Delete processing message
                await processing_msg.delete()
                
                # Send the generated image
                caption = (
                    f"🖼️ **Your AI-Generated Image**\n\n"
                    f"📝 _{prompt[:200]}_\n\n"
                    f"🤖 Generated by @ArtForgeBot\n"
                    f"⚡ Powered by {', '.join(Config.get_active_providers())}"
                )
                
                await update.message.reply_photo(
                    photo=io.BytesIO(image_data),
                    caption=caption,
                    parse_mode='Markdown'
                )
            else:
                await processing_msg.edit_text(
                    "❌ **Generation Failed**\n\n"
                    "I couldn't generate an image from your prompt. Please try:\n"
                    "• Using a different description\n"
                    "• Making the prompt more specific\n"
                    "• Trying again later\n\n"
                    "💡 Check /examples for inspiration!"
                )
                
        except asyncio.TimeoutError:
            await processing_msg.edit_text(
                "⏰ **Timeout Error**\n\n"
                "Image generation is taking too long. Please try again with a simpler prompt."
            )
        except Exception as e:
            logger.error(f"Error in _process_generation: {str(e)}")
            logger.error(traceback.format_exc())
            await processing_msg.edit_text(
                "❌ **Error Occurred**\n\n"
                "Something went wrong while generating your image.\n"
                "Please try again later."
            )
    
    @staticmethod
    async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle inline button callbacks"""
        query = update.callback_query
        await query.answer()
        
        if query.data == "generate":
            await query.edit_message_text(
                "🎨 **Send me your image description!**\n\n"
                "Just type what you want to see and I'll create it.\n\n"
                "💡 Example: 'A beautiful castle on a hill at sunset'"
            )
        elif query.data == "help":
            await BotHandlers.help(update, context)
        elif query.data == "status":
            await BotHandlers.status(update, context)
        elif query.data == "examples":
            await BotHandlers.examples(update, context)

# ============================================
# MAIN APPLICATION
# ============================================

def main() -> None:
    """Main entry point for the bot"""
    try:
        logger.info("=" * 50)
        logger.info("🚀 Starting ArtForgeBot...")
        logger.info(f"🤖 Bot: @ArtForgeBot")
        logger.info(f"📊 Providers: {', '.join(Config.get_active_providers())}")
        
        if not Config.is_configured():
            logger.warning("⚠️ No AI providers configured!")
        
        # Create application
        application = Application.builder().token(Config.BOT_TOKEN).build()
        
        # Register command handlers
        application.add_handler(CommandHandler("start", BotHandlers.start))
        application.add_handler(CommandHandler("help", BotHandlers.help))
        application.add_handler(CommandHandler("status", BotHandlers.status))
        application.add_handler(CommandHandler("examples", BotHandlers.examples))
        application.add_handler(CommandHandler("generate", BotHandlers.generate))
        
        # Register message handler for text messages
        application.add_handler(
            MessageHandler(
                filters.TEXT & ~filters.COMMAND,
                BotHandlers.handle_message
            )
        )
        
        # Register callback handler for inline buttons
        application.add_handler(CallbackQueryHandler(BotHandlers.button_callback))
        
        # Start the bot
        logger.info("✅ Bot is running! Press Ctrl+C to stop.")
        logger.info("=" * 50)
        
        # Run the bot
        application.run_polling(
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=True
        )
        
    except KeyboardInterrupt:
        logger.info("🛑 Bot stopped by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
