# eval/test_set.py
# 20 questions with ground truth answers and expected source type.
# Used by all evaluation scripts below.

TEST_SET = [
    # --- Exercise Agent (7 questions) ---
    {
        "id": 1,
        "question": "What are some good chest exercises I can do with a barbell?",
        "ground_truth": "Barbell bench press, incline barbell press, and barbell floor press are effective chest exercises using a barbell.",
        "source_type": "exercise",
        "expected_route": "exercise",
    },
    {
        "id": 2,
        "question": "Give me beginner-friendly back exercises.",
        "ground_truth": "Beginner back exercises include lat pulldowns, seated cable rows, and dumbbell bent-over rows.",
        "source_type": "exercise",
        "expected_route": "exercise",
    },
    {
        "id": 3,
        "question": "What exercises target the quadriceps?",
        "ground_truth": "Squats, leg press, and lunges are primary quadriceps exercises.",
        "source_type": "exercise",
        "expected_route": "exercise",
    },
    {
        "id": 4,
        "question": "What are bodyweight exercises for shoulders?",
        "ground_truth": "Pike push-ups, handstand push-ups, and lateral raises with resistance bands target the shoulders without equipment.",
        "source_type": "exercise",
        "expected_route": "exercise",
    },
    {
        "id": 5,
        "question": "Show me tricep exercises I can do with a cable machine.",
        "ground_truth": "Cable tricep pushdowns, overhead cable tricep extensions, and cable kickbacks are effective cable tricep exercises.",
        "source_type": "exercise",
        "expected_route": "exercise",
    },
    {
        "id": 6,
        "question": "What are common mistakes in the deadlift?",
        "ground_truth": "Common deadlift mistakes include rounding the lower back, jerking the bar, and letting the bar drift away from the body.",
        "source_type": "exercise",
        "expected_route": "exercise",
    },
    {
        "id": 7,
        "question": "What ab exercises can I do at home with no equipment?",
        "ground_truth": "Planks, crunches, bicycle crunches, and leg raises are effective bodyweight ab exercises.",
        "source_type": "exercise",
        "expected_route": "exercise",
    },

    # --- Nutrition Agent (7 questions) ---
    {
        "id": 8,
        "question": "How much protein should I eat per day to build muscle?",
        "ground_truth": "For muscle gain, aim for 1.6 to 2.2 grams of protein per kilogram of body weight per day.",
        "source_type": "nutrition",
        "expected_route": "nutrition",
    },
    {
        "id": 9,
        "question": "What should I eat before a workout?",
        "ground_truth": "A pre-workout meal should include easily digestible carbohydrates and moderate protein, eaten 1-2 hours before training.",
        "source_type": "nutrition",
        "expected_route": "nutrition",
    },
    {
        "id": 10,
        "question": "What should I eat after a workout?",
        "ground_truth": "Post-workout nutrition should include protein to support muscle repair and carbohydrates to replenish glycogen, ideally within 30-60 minutes.",
        "source_type": "nutrition",
        "expected_route": "nutrition",
    },
    {
        "id": 11,
        "question": "How much water should I drink during training?",
        "ground_truth": "Drink 500ml before exercise, 150-250ml every 15-20 minutes during, and rehydrate after based on sweat loss.",
        "source_type": "nutrition",
        "expected_route": "nutrition",
    },
    {
        "id": 12,
        "question": "What are the best supplements for strength training?",
        "ground_truth": "Creatine monohydrate, protein powder, and caffeine are the most evidence-backed supplements for strength training.",
        "source_type": "nutrition",
        "expected_route": "nutrition",
    },
    {
        "id": 13,
        "question": "How do I calculate my daily calorie intake for fat loss?",
        "ground_truth": "Calculate your TDEE then apply a caloric deficit of 300-500 kcal per day for sustainable fat loss.",
        "source_type": "nutrition",
        "expected_route": "nutrition",
    },
    {
        "id": 14,
        "question": "What are good high-protein meal prep ideas?",
        "ground_truth": "Grilled chicken with rice, Greek yogurt, boiled eggs, and lentil soup are high-protein meal prep staples.",
        "source_type": "nutrition",
        "expected_route": "nutrition",
    },

    # --- Program Builder Agent (4 questions) ---
    {
        "id": 15,
        "question": "Build me a 3-day beginner full body workout program.",
        "ground_truth": "A 3-day beginner full body program includes compound movements like squats, bench press, and rows spread across Monday, Wednesday, and Friday.",
        "source_type": "program",
        "expected_route": "program",
    },
    {
        "id": 16,
        "question": "I want a 4-day upper lower split for intermediate lifters.",
        "ground_truth": "An upper/lower split trains upper body twice and lower body twice per week, rotating through compound and accessory movements.",
        "source_type": "program",
        "expected_route": "program",
    },
    {
        "id": 17,
        "question": "Give me a 5-day push pull legs program for muscle gain.",
        "ground_truth": "A PPL program trains push muscles (chest, shoulders, triceps), pull muscles (back, biceps), and legs on alternating days across 5-6 sessions per week.",
        "source_type": "program",
        "expected_route": "program",
    },
    {
        "id": 18,
        "question": "Create a home workout plan using only dumbbells, 3 days per week.",
        "ground_truth": "A 3-day dumbbell home program covers full body or push/pull/legs splits using dumbbell variations of major compound movements.",
        "source_type": "program",
        "expected_route": "program",
    },

    # --- Mixed / Edge cases (2 questions) ---
    {
        "id": 19,
        "question": "I want to lose weight. What should I eat and what exercises should I do?",
        "ground_truth": "For fat loss, maintain a caloric deficit with high protein intake and combine resistance training with cardio.",
        "source_type": "mixed",
        "expected_route": "nutrition+exercise",
    },
    {
        "id": 20,
        "question": "I am a complete beginner. Where do I start?",
        "ground_truth": "Beginners should start with a 3-day full body program focusing on compound lifts, adequate protein intake, and progressive overload.",
        "source_type": "mixed",
        "expected_route": "program",
    },
]
