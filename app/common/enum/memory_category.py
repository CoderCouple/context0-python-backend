"""Memory Category Tags for organizing memories by domain"""

from enum import Enum
from typing import List, Dict


class MemoryCategory(str, Enum):
    """Predefined memory categories for consistent tagging"""

    # Personal Life
    PERSONAL = "personal"
    FAMILY = "family"
    RELATIONSHIPS = "relationships"
    HEALTH = "health"
    FITNESS = "fitness"

    # Preferences
    PREFERENCE = "preference"
    LIKES = "likes"
    DISLIKES = "dislikes"
    FAVORITES = "favorites"

    # Food & Lifestyle
    FOOD = "food"
    COOKING = "cooking"
    RESTAURANTS = "restaurants"
    DIET = "diet"

    # Education & Learning
    EDUCATION = "education"
    LEARNING = "learning"
    SKILLS = "skills"
    COURSES = "courses"
    RESEARCH = "research"

    # Work & Career
    WORK = "work"
    CAREER = "career"
    PROJECTS = "projects"
    MEETINGS = "meetings"
    GOALS = "goals"

    # Entertainment & Hobbies
    ENTERTAINMENT = "entertainment"
    HOBBIES = "hobbies"
    MOVIES = "movies"
    MUSIC = "music"
    BOOKS = "books"
    GAMES = "games"
    SPORTS = "sports"

    # Travel & Places
    TRAVEL = "travel"
    PLACES = "places"
    VACATION = "vacation"
    LOCATIONS = "locations"

    # Finance & Shopping
    FINANCE = "finance"
    BUDGET = "budget"
    SHOPPING = "shopping"
    INVESTMENTS = "investments"

    # Technology
    TECHNOLOGY = "technology"
    CODING = "coding"
    TOOLS = "tools"
    APPS = "apps"

    # Social & Events
    SOCIAL = "social"
    EVENTS = "events"
    CELEBRATIONS = "celebrations"
    NETWORKING = "networking"

    # Ideas & Creativity
    IDEAS = "ideas"
    CREATIVITY = "creativity"
    INSPIRATION = "inspiration"
    BRAINSTORMING = "brainstorming"

    # Tasks & Productivity
    TASKS = "tasks"
    TODOS = "todos"
    REMINDERS = "reminders"
    DEADLINES = "deadlines"

    # General
    GENERAL = "general"
    MISCELLANEOUS = "miscellaneous"
    NOTES = "notes"


# Category groupings for better organization
CATEGORY_GROUPS: Dict[str, List[MemoryCategory]] = {
    "Personal Life": [
        MemoryCategory.PERSONAL,
        MemoryCategory.FAMILY,
        MemoryCategory.RELATIONSHIPS,
        MemoryCategory.HEALTH,
        MemoryCategory.FITNESS,
    ],
    "Preferences": [
        MemoryCategory.PREFERENCE,
        MemoryCategory.LIKES,
        MemoryCategory.DISLIKES,
        MemoryCategory.FAVORITES,
    ],
    "Food & Lifestyle": [
        MemoryCategory.FOOD,
        MemoryCategory.COOKING,
        MemoryCategory.RESTAURANTS,
        MemoryCategory.DIET,
    ],
    "Education & Learning": [
        MemoryCategory.EDUCATION,
        MemoryCategory.LEARNING,
        MemoryCategory.SKILLS,
        MemoryCategory.COURSES,
        MemoryCategory.RESEARCH,
    ],
    "Work & Career": [
        MemoryCategory.WORK,
        MemoryCategory.CAREER,
        MemoryCategory.PROJECTS,
        MemoryCategory.MEETINGS,
        MemoryCategory.GOALS,
    ],
    "Entertainment & Hobbies": [
        MemoryCategory.ENTERTAINMENT,
        MemoryCategory.HOBBIES,
        MemoryCategory.MOVIES,
        MemoryCategory.MUSIC,
        MemoryCategory.BOOKS,
        MemoryCategory.GAMES,
        MemoryCategory.SPORTS,
    ],
    "Travel & Places": [
        MemoryCategory.TRAVEL,
        MemoryCategory.PLACES,
        MemoryCategory.VACATION,
        MemoryCategory.LOCATIONS,
    ],
    "Finance & Shopping": [
        MemoryCategory.FINANCE,
        MemoryCategory.BUDGET,
        MemoryCategory.SHOPPING,
        MemoryCategory.INVESTMENTS,
    ],
    "Technology": [
        MemoryCategory.TECHNOLOGY,
        MemoryCategory.CODING,
        MemoryCategory.TOOLS,
        MemoryCategory.APPS,
    ],
    "Social & Events": [
        MemoryCategory.SOCIAL,
        MemoryCategory.EVENTS,
        MemoryCategory.CELEBRATIONS,
        MemoryCategory.NETWORKING,
    ],
    "Ideas & Creativity": [
        MemoryCategory.IDEAS,
        MemoryCategory.CREATIVITY,
        MemoryCategory.INSPIRATION,
        MemoryCategory.BRAINSTORMING,
    ],
    "Tasks & Productivity": [
        MemoryCategory.TASKS,
        MemoryCategory.TODOS,
        MemoryCategory.REMINDERS,
        MemoryCategory.DEADLINES,
    ],
}


# Keywords mapping for auto-categorization
CATEGORY_KEYWORDS: Dict[MemoryCategory, List[str]] = {
    MemoryCategory.FOOD: [
        "eat",
        "meal",
        "breakfast",
        "lunch",
        "dinner",
        "snack",
        "recipe",
        "dish",
        "cuisine",
        "hungry",
    ],
    MemoryCategory.EDUCATION: [
        "learn",
        "study",
        "course",
        "class",
        "school",
        "university",
        "degree",
        "certification",
        "training",
    ],
    MemoryCategory.PERSONAL: [
        "I",
        "me",
        "my",
        "feel",
        "think",
        "believe",
        "personal",
        "private",
    ],
    MemoryCategory.PREFERENCE: [
        "like",
        "prefer",
        "love",
        "hate",
        "enjoy",
        "favorite",
        "best",
        "worst",
    ],
    MemoryCategory.HEALTH: [
        "doctor",
        "medicine",
        "sick",
        "healthy",
        "exercise",
        "symptoms",
        "treatment",
        "therapy",
    ],
    MemoryCategory.WORK: [
        "job",
        "office",
        "colleague",
        "boss",
        "meeting",
        "project",
        "deadline",
        "task",
        "client",
    ],
    MemoryCategory.TRAVEL: [
        "trip",
        "vacation",
        "flight",
        "hotel",
        "visit",
        "tourist",
        "destination",
        "journey",
    ],
    MemoryCategory.FINANCE: [
        "money",
        "budget",
        "expense",
        "income",
        "save",
        "invest",
        "pay",
        "cost",
        "price",
    ],
    MemoryCategory.FAMILY: [
        "family",
        "mother",
        "father",
        "sister",
        "brother",
        "parent",
        "child",
        "spouse",
        "relative",
    ],
    MemoryCategory.ENTERTAINMENT: [
        "watch",
        "movie",
        "show",
        "series",
        "game",
        "play",
        "fun",
        "enjoy",
        "entertainment",
    ],
}


def suggest_categories(text: str) -> List[MemoryCategory]:
    """Suggest relevant categories based on text content"""
    text_lower = text.lower()
    suggested = set()

    # Check keywords
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in text_lower for keyword in keywords):
            suggested.add(category)

    # If no suggestions, return GENERAL
    if not suggested:
        suggested.add(MemoryCategory.GENERAL)

    return list(suggested)
