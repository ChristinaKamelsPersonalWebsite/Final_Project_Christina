from tools.exercise_tools import filter_exercises

results = filter_exercises(
    primary_muscle="chest",
    equipment="dumbbell",
    difficulty="beginner"
)

print("Results:", results[:3])