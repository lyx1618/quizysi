import discord
from discord.ext import commands
import json
import logging
from datetime import datetime
import pytz
import signal  # For handling graceful shutdown
import re  # Regex for detecting custom emoji codes
import asyncio
import pandas as pd
import random
from collections import Counter
from dotenv import load_dotenv
import os

load_dotenv()
BOT_TOKEN = os.getenv('BOT_TOKEN')

# Setup logging
logging.basicConfig(level=logging.INFO)

# Bot setup
intents = discord.Intents.default()
intents.message_content = True  # Enable message content intent
intents.members = True  # Enable members intent if you need member data like joins/roles
bot = commands.Bot(command_prefix="!", intents=intents)

# Time zone setup (Eastern Standard Time)
est_tz = pytz.timezone("US/Eastern")

# Data initialization (in-memory dictionary structure)
season_points = {}  # Points for the current season
total_points = {}  # Cumulative points across all seasons
historical_seasons = {}  # Store historical seasons' data

last_reset = {"season": 1, "timestamp": datetime.utcnow()}  # Track season and last reset time

current_window_users = {"7:27 AM": [], "7:27 PM": []}  # Track users for each time window

# Track 727 at all times
track_727_always = False

# Load data
material = []
with open("/home/container/luke_plain", "r") as file:
    info = file.read()

verse = info.splitlines()
for x in verse:

    verse_ref = x.split(":",3)
    verse_ref = [int(x) if x.isdigit() else x for x in verse_ref]
    
    quote = verse_ref[3].replace('\xa0', '').strip()
    newverse = dict({"book": verse_ref[0], "chapter": verse_ref[1], "verse": verse_ref[2], "quote": quote})
    if material.count(newverse) < 1:
        material.append(newverse)
    else:
        break

df = pd.DataFrame(material)
pd.set_option('display.max_colwidth', None)

# Custom JSON Encoder to handle datetime objects
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()  # Convert datetime to string (ISO format)
        return super().default(obj)  # Use the default encoder for other types

def save_data(season_points, total_points, last_reset, historical_seasons):
    """Save bot data to a file."""
    data_to_save = {
        "season_points": season_points,
        "total_points": total_points,
        "last_reset": {"season": last_reset["season"], "timestamp": last_reset["timestamp"]},
        "historical_seasons": historical_seasons
    }

    with open("/home/container/bot_data.json", "w") as f:
        json.dump(data_to_save, f, indent=4, cls=CustomJSONEncoder)

# Ensure user-specific data is tracked
def track_user(user):
    """Ensure user data is user-specific."""
    user_id = str(user.id)
    if user_id not in season_points:
        season_points[user_id] = {"points": 0, "username": user.name}
    if user_id not in total_points:
        total_points[user_id] = {"points": 0, "username": user.name}

@bot.event
async def on_ready():
    logging.info(f'Logged in as {bot.user}')

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    track_user(message.author)

    current_time = datetime.now(pytz.utc).astimezone(est_tz).strftime('%I:%M %p')
    is_727_time = current_time in ["7:27 AM", "7:27 PM"]

    # Check if the message contains "727" or a custom emoji with the code 727
    if "727" in message.content or re.search(r"<:[a-zA-Z0-9_]+:727>", message.content):
        if track_727_always:
            # If track_727_always is enabled, give 1 point for each instance of 727
            season_points[str(message.author.id)]["points"] += 1
            total_points[str(message.author.id)]["points"] += 1
            logging.info(f"User {message.author} typed '727'. Awarded 1 point (track_727_always enabled).")
        elif is_727_time:
            # If track_727_always is disabled and it's 7:27 AM/PM, use the 2, 1, 0 points system
            window_key = "7:27 AM" if current_time == "7:27 AM" else "7:27 PM"

            # Add the user to the current window if they haven't already triggered 727
            if message.author.id not in current_window_users[window_key]:
                current_window_users[window_key].append(message.author.id)

                # Assign points based on their position in the list
                user_list = current_window_users[window_key]
                user_index = user_list.index(message.author.id)

                if user_index == 0:
                    # First user gets 2 points
                    points = 2
                elif user_index == len(user_list) - 1:
                    # Last user gets 0 points
                    points = 0
                else:
                    # Users in between get 1 point
                    points = 1

                # Update the user's points
                season_points[str(message.author.id)]["points"] += points
                total_points[str(message.author.id)]["points"] += points
                logging.info(f"User {message.author} got {points} points for being in position {user_index + 1} in {window_key}.")

    await bot.process_commands(message)

# Find verse based on reference
def versequery(book, chapter, verse):
    quote = df[(df['book'] == book) & (df['chapter'] == chapter) & (df['verse'] == verse)]
    quote = quote[['quote']]
    quote = quote.to_string(index=False, header=False)
    return quote + '\n(' + book + ' ' + str(chapter) + ':' + str(verse) + ')'

# Find verse based on keywords
def wordquery(words, limit = 10):
    if not words:
        return f"https://tenor.com/view/think-meme-thinking-memes-memes-2024-gif-6703217797690493255"
    
    # If `words is a single string, split it into a list of words
    if isinstance(words, str):
        words = words.split()
    
    # Create a regex pattern to match the entire phrase (all words together)
    phrase_pattern = r'\b' + re.escape(" ".join(words)) + r'\b'
    
    # Find rows where the quote contains the entire phrase (case-insensitive)
    matches = df[df['quote'].str.contains(phrase_pattern, case=False, regex=True)]
    
    # If no matches are found, return a message
    if matches.empty:
        return f"No quotes found containing the phrase: '{' '.join(words)}'"
    
    # Extract book, chapter, and verse information for the matching rows
    to_query = matches[['book', 'chapter', 'verse']].values.tolist()
    
    # Query and collect results for each match (up to the specified limit)
    results = []
    for i, values in enumerate(to_query):
        if i >= limit:  # Stop after reaching the limit
            return f"Way too many"
        book, chapter, verse = values
        result = versequery(book, chapter, verse)
        
        # Bold the entire phrase in the result (case-insensitive)
        result = re.sub(phrase_pattern, r"**\g<0>**", result, flags=re.IGNORECASE)
        results.append(result)
    
    # Return all results as a single string
    return "\n\n".join(results)
    
# Create MA question
def multans():
    # MA docs link
    url = f"https://docs.google.com/spreadsheets/d/e/2PACX-1vSeZ1WU8cIu_viDmGGMypjDOaollcERBgvvVfGyeblh_bLLZ5Lagi8TMn-4t1De8sch2LGY2S99K119/pub?gid=1359126200&single=true&output=csv"
    
    # Create a list of questions
    ma_list = pd.read_csv(url)
    pool = ma_list["Question"].tolist()
    
    # Select a random index
    seed = random.randint(1,len(ma_list) - 1)
    select_prompt = pool[seed].split('>>')
    ans = ma_list.iat[seed,2]
    
    # TEMPORARY CODE
    ref = ma_list.iat[seed,0].replace(' ',':').split(':')
    ref = [int(x) if x.isdigit() else x for x in ref]
    book, chapter, verse = ref
    book = 'Luke'

    # Generate prompt
    prompt = select_prompt[0] + f"||{select_prompt[1]}||"
    prompt += f"\n\n**Answer:**\n||{ans}||\n\n**Verse:**\n||{versequery(book,chapter,verse)}||"
    
    return prompt
    
# Find a random one word key
def keyword1():
    # Split the text into words
    combined_text = df['quote'].str.cat(sep=' ')
    combined_text = re.sub(r'—', ' ', combined_text)   
    preparse = combined_text.split()
    
    # Clean punctuation 
    pattern = r'[—,!?;:.“‘’”\()]' 
    cleaned_words = [re.sub(pattern, '', word) for word in preparse]
    
    # Normalize case for counting
    lower_words = [word.lower() for word in cleaned_words]
    
    # Count word occurrences
    word_counts = Counter(lower_words)
    
    # Filter words that appear only once
    unique_lower_words = {word for word, count in word_counts.items() if count == 1}
    
    # Return the original words that are unique
    pool = [word for word in cleaned_words if word.lower() in unique_lower_words]
    pool.pop(-40)
    
    return pool
    
# Find a two word key
def keyword2():
    # Combine data to string
    combined_text = df['quote'].str.cat(sep=' ')
    combined_text = re.sub(r'—', ' ', combined_text)

    # Iterate through each keyword and replace it with an placeholder
    pattern = r'\b(?:' + '|'.join(map(re.escape, keyword1())) + r')\b'
    result = re.sub(pattern, '&&', combined_text, flags=re.IGNORECASE).split()
    
    # combine all words into phrases of two words and replace phrases with breaks in flow with a placeholder
    phrase = [' '.join(result[x:x+2]) for x in range(len(result)-1)]
    clean_phrase = [re.sub(r'\b(\w+)([^\w\s]+\s+)(\w+)\b', '&&', word) for word in phrase]
    
    # Remove placeholders from the list and clean punctuation
    remove_keys = pd.Series(clean_phrase)
    matches = remove_keys[~remove_keys.str.contains('&&')]
    no_keys = matches.values.tolist()
    pattern = r'[—,!?;:.“‘’”\()]' 
    clean_punc = [re.sub(pattern, '', word) for word in no_keys]

    # Find all phrases that only occur once
    lower_words = [word.lower() for word in clean_punc]    
    word_counts = Counter(lower_words)
    unique_lower_words = {word for word, count in word_counts.items() if count == 1}
    pool = [word for word in clean_punc if word.lower() in unique_lower_words]

    return pool

# Find a three word key
def keyword3():
    # Combine data to string
    combined_text = df['quote'].str.cat(sep=' ')
    combined_text = re.sub(r'—', ' ', combined_text)
    
    # Iterate through each keyword and replace it with an placeholder
    pattern1 = r'\b(?:' + '|'.join(map(re.escape, keyword1())) + r')\b'
    pattern2 = r'\b(?:' + '|'.join(map(re.escape, keyword2())) + r')\b'
    result = re.sub(pattern1, '&&', combined_text, flags=re.IGNORECASE)
    result = re.sub(pattern2, '&&', result, flags=re.IGNORECASE).split()
    
    # combine all words into phrases of three words and replace phrases with breaks in flow with a placeholder
    phrase = [' '.join(result[x:x+3]) for x in range(len(result)-2)]
    clean_phrase = [re.sub(r'[^\w\s\'\-]', '&&', word) for word in phrase]

    # Remove placeholders from the list and clean punctuation
    remove_keys = pd.Series(clean_phrase)
    matches = remove_keys[~remove_keys.str.contains('&&')]
    no_keys = matches.values.tolist()
    pattern = r'[—,!?;:.“‘’”\()]' 
    clean_punc = [re.sub(pattern, '', word) for word in no_keys]
    
    
    phrase_to_original = {phrase.lower(): phrase for phrase in clean_punc}
    phrase_counts = Counter(phrase.lower() for phrase in clean_punc)
    pool = [phrase_to_original[phrase] for phrase, count in phrase_counts.items() if count == 1]
 
    return pool

async def create_or_update_role(user, role_name, color):
    """Create or update the role and assign it to the user."""
    role = discord.utils.get(user.guild.roles, name=role_name)
    if not role:
        # Create new role if it doesn't exist
        role = await user.guild.create_role(name=role_name, color=color, reason="Top Season Role")
    else:
        # Update the color of the existing role if necessary
        if role.color != color:
            await role.edit(color=color)
    await user.add_roles(role)
    logging.info(f"Assigned {role_name} role to {user}.")

async def assign_season_roles(sorted_season_users):
    """Assign roles based on users' ranking in the season."""
    guild = bot.get_guild(730217496791220244)  # Add your guild ID here
    if not guild:
        logging.error("Guild not found.")
        return

    for idx, (user_id, points) in enumerate(sorted_season_users):
        try:
            user = await guild.fetch_member(user_id)  # Use fetch_member to ensure the user is fetched
            if user:
                # Assign role based on ranking
                placement = idx + 1
                role_name = f"Season_S{last_reset['season']}_#{placement}"

                # Assign color based on ranking
                color = discord.Color.red() if idx == 0 else discord.Color.gold() if idx == 1 else discord.Color.from_rgb(192, 192, 192) if idx == 2 else discord.Color.from_rgb(139, 69, 19) if idx == 3 else discord.Color.dark_grey()

                # Assign role
                await create_or_update_role(user, role_name, color)

                # Log the role assignment
                logging.info(f"Assigned {role_name} role to {user} for season {last_reset['season']}.")
            else:
                logging.error(f"User {user_id} not found in the guild.")

        except Exception as e:
            logging.error(f"Error assigning role to user {user_id}: {e}")

    # Reassign the 'TopTotal' role after assigning season roles
    await assign_toptotal_role()

async def assign_toptotal_role():
    """Reassign the 'mrekk' role to the user with the most total points."""
    guild = bot.get_guild(730217496791220244)  # Add your guild ID here
    if not guild:
        logging.error("Guild not found.")
        return
    if total_points:
        # Get the user with the most points
        top_user_id = max(total_points, key=lambda x: total_points[x]['points'])
        top_user = await guild.fetch_member(top_user_id)  # Use fetch_member to get a Member object
        if top_user:
            # Fetch the 'mrekk' role
            role = discord.utils.get(top_user.guild.roles, name="mrekk")

            # If the role exists, remove it from all members who had it
            if role:
                # Remove the 'mrekk' role from all members who currently have it
                for member in top_user.guild.members:
                    if role in member.roles:
                        await member.remove_roles(role)
                        logging.info(f"Removed 'mrekk' role from {member.name}")

            else:
                # Create the role if it doesn't exist
                role = await top_user.guild.create_role(name="mrekk", color=discord.Color.blue())
                logging.info("Created 'mrekk' role.")

            # Assign the 'mrekk' role to the user with the most points
            await top_user.add_roles(role)
            logging.info(f"Assigned 'mrekk' role to {top_user}.")

@bot.command()
async def leaderboard(ctx):
    current_season = sorted(season_points.items(), key=lambda x: x[1]['points'], reverse=True)
    all_time = sorted(total_points.items(), key=lambda x: x[1]['points'], reverse=True)
    response = "**Current Season Leaderboard**\n"
    for idx, (user_id, data) in enumerate(current_season, 1):
        response += f"{idx}. {data['username']} - {data['points']} points\n"
    response += "\n**All-Time Leaderboard**\n"
    for idx, (user_id, data) in enumerate(all_time, 1):
        response += f"{idx}. {data['username']} - {data['points']} points\n"
    await ctx.send(response)

@bot.command()
async def aslan(ctx):
    await ctx.send("imma crash out")

@bot.command()
async def alex(ctx):
    await ctx.send("go to sleep")

@bot.command()
async def andy(ctx):
    await ctx.send("Who put streams in my jump map")

@bot.command()
async def spark(ctx):
    await ctx.send("370")
    
@bot.command()
async def toggle(ctx):  
    global track_727_always
    track_727_always = not track_727_always
    status = "enabled" if track_727_always else "disabled"
    await ctx.send(f"727 tracking is now {status}.")
    logging.info(f"727 tracking {status} by {ctx.author}")
    
@bot.command()
async def kw(ctx, n=0):
    if n == 0:
        n = random.randint(1,3)
    if n == 1:
        prompt = random.choice(keyword1())
    elif n == 2:
        prompt = random.choice(keyword2())
    elif n == 3:
        prompt = random.choice(keyword3())
    else: 
        await ctx.send(f"https://tenor.com/view/nuh-uh-beocord-no-lol-gif-24435520")
        return
    
    prompt += f"\n\n||{wordquery(prompt)}||\n"
    await ctx.send(prompt)

@bot.command()
async def f(ctx, *, word=""):
    await ctx.send(wordquery(word))
    
@bot.command()
async def ma(ctx):
    await ctx.send(multans())
    
@bot.command()
@commands.has_permissions(administrator=True)
async def factoryreset(ctx):
    """
    Administrator command to reset all the points.
    """
    global season_points, last_reset

    # Confirm the reset with the administrator
    confirmation_message = await ctx.send("Are you sure you want to factory reset? This action cannot be undone. Type `yes` to confirm.")

    def check(m):
        return m.author == ctx.author and m.channel == ctx.channel and m.content.lower() == "yes"

    try:
        # Wait for the administrator to confirm
        await bot.wait_for("message", timeout=30.0, check=check)
    except asyncio.TimeoutError:
        # If no confirmation is received within 30 seconds, cancel the reset
        await ctx.send("Season reset cancelled.")
        return

    # Reset the season points and increment the season number
    season_points.clear()
    total_points.clear()

    # Notify the administrator
    await ctx.send(f"Factory reset complete")
    logging.info(f"Season reset to S{last_reset['season']} by {ctx.author}.")
    
@bot.command()
@commands.has_permissions(administrator=True)
async def resetpoints(ctx):
    """
    Administrator command to reset the current season points.
    """
    global season_points, last_reset

    # Confirm the reset with the administrator
    confirmation_message = await ctx.send("Are you sure you want to reset the current season points? This action cannot be undone. Type `yes` to confirm.")

    def check(m):
        return m.author == ctx.author and m.channel == ctx.channel and m.content.lower() == "yes"

    try:
        # Wait for the administrator to confirm
        await bot.wait_for("message", timeout=30.0, check=check)
    except asyncio.TimeoutError:
        # If no confirmation is received within 30 seconds, cancel the reset
        await ctx.send("Season reset cancelled.")
        return

    # Reset the season points and increment the season number
    season_points.clear()

    # Notify the administrator
    await ctx.send(f"Season points have been reset")
    logging.info(f"Season reset to S{last_reset['season']} by {ctx.author}.")
    
@bot.command()
@commands.has_permissions(administrator=True)
async def endseason(ctx):
    global last_reset
    
    # Confirm the reset with the administrator
    confirmation_message = await ctx.send("Are you sure you want to reset the current season points? This action cannot be undone. Type `yes` to confirm.")

    def check(m):
        return m.author == ctx.author and m.channel == ctx.channel and m.content.lower() == "yes"

    try:
        # Wait for the administrator to confirm
        await bot.wait_for("message", timeout=30.0, check=check)
    except asyncio.TimeoutError:
        # If no confirmation is received within 30 seconds, cancel the reset
        await ctx.send("Season reset cancelled.")
        return
    
    sorted_season_users = sorted(season_points.items(), key=lambda x: x[1]['points'], reverse=True)
    historical_seasons[last_reset['season']] = season_points
    await assign_season_roles(sorted_season_users)
    
    season_points.clear()
    last_reset['season'] += 1
    logging.info(f"Season reset to S{last_reset['season']}.")

    await ctx.send(f"Season has ended and leaderboard updated.")

def shutdown_signal_handler(signal, frame):
    save_data(season_points, total_points, last_reset, historical_seasons)
    logging.info("Data saved. Bot is shutting down.")
    asyncio.get_event_loop().stop()

signal.signal(signal.SIGINT, shutdown_signal_handler)
signal.signal(signal.SIGTERM, shutdown_signal_handler)

# Run the bot with your token
bot.run(BOT_TOKEN)
