# Food Tracker
**Parsing of the Meal Genie is jank...I'm working on it**

**This is a super simple and made to be run locally. I initially made it bespoke to my environment, but found it useful so I decided to remove hard-coding and post; however, it is still not a "baked" project so it may require tweaking**

## Features
- Enter your personal metric targets
    - First entry may require saving the JSON to food-tracker/docs/data/metrics.json
- Add and Edit food/ingredients and Recipes
    - LLM assistance with new foods
    - Recipe entry with instructions, notes and ratings
    - (recipes are aggregates of existing ingredients)
- Track and Edit intake (daily and timestamp)
    - Individual ingredients/foods
    - Servings of Recipes
    - Change the Date to view past tracked entries
- See daily results
    - Totals
    - % against your Targets
    - Calorie trends throughout the day
- Day Planner (MVP) - I Left it in even after it was updated for thoise who want more simplicity
    - Plan a day of meals and snacks to see results
    - See basic suggestions and substitution ideas
    - Suggestions are hard-coded sort of "if-then" responses
- Planning Assistant (Heavy LLM)
    - Plan a day of meals and snacks to see results
    - Includes Suggestions and Substitution Ideas
    - Has a simple chatbot
    - Meal Genie will take your metrics and added requirements to:
        - Build a 3-day menu of meals and snacks
        - Build a grocery list
- Reporting 
    - Run reports to visualize
        - Totals and % against target across a range
        - Calorie source breakdown
        - Hourly trends with some hard-coded notes
        - Rolling average and trends for each macro
        - JSON Export 

**If there are specfic notes you want to always be considered for your diet then simply edit the LLM prompts in app.py and relaunch the server**

## Installation and Setup

### Notes
**I used the OpenAI API for any LLM use. If you want the same, update .env with your OPENAI_API_KEY, or the key to your LLM API of choice, or spin up your own locally then edit backend app.py accordingly**
```python
# NOTE: If you point OPENAI_BASE_URL at a local/OpenAI-compatible service,
# this app still expects OPENAI_API_KEY to be set. Use a dummy value if your
# provider ignores it.
load_dotenv("../.env")  
load_dotenv()           # also allow .env in the current working dir

LLM_PROVIDER = (os.getenv("LLM_PROVIDER") or "openai").strip().lower()
OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()
OPENAI_BASE_URL = (os.getenv("OPENAI_BASE_URL") or "").strip()
```

**I added Reporting functionality in at the end so you'll notice it is relatively isolated from the main pages so as not to break what already worked. Feel free to integrate reports.js and tools/reports_api_parquet.py if you like otherwise they are referenced/imported instead**

**Edit app.py to use your IP Address or localhost if you do not wat to bind to all**
```python
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
```

### Directory Structure
```
[your projects]\FOOD-TRACKER
    │   .env
    │   README.md
    │
    └───docs
        │   add.html
        │   app.py
        │   index.html
        │   plan.html
        │   plan_bot.html
        │   reports.html
        │   reports.js
        │   results.html
        │   style.css
        │   track.html
        │
        ├───data
        │       calendar.parquet
        │       ingredients.parquet
        │       metrics.json
        │       recipes.parquet
        │
        └───tools
                reports_api_parquet.py
                __init__.py
```
## Running
**Just run from the docs/ directory from CLI**

```
EXAMPLE
cd /Projects/food-tracker/docs
python app.py
```

