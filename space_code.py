#!/usr/bin/env python3
"""
GAIA Benchmark Submission - Generated Space Code
This file contains the model answers for GAIA benchmark evaluation.
"""

# GAIA submission answers in the required format
ANSWERS = [
    {
        "task_id": "8e867cd7-cff9-4e6c-867a-ff5ddc2550be",
        "submitted_answer": '3'
    },
    {
        "task_id": "a1e91b78-d3d8-4675-bb8d-62741b4b68a6",
        "submitted_answer": '3'
    },
    {
        "task_id": "2d83110e-a098-4ebb-9987-066c06fa42d0",
        "submitted_answer": 'right'
    },
    {
        "task_id": "cca530fc-4052-43b2-b130-b30968d8aa44",
        "submitted_answer": 'Qe1+'
    },
    {
        "task_id": "4fc2f1ae-8625-45b5-ab34-ad4433bc21f8",
        "submitted_answer": '• * The instructions require the answer to be "a number OR as few words as possible OR a comma separated list". Since I cannot provide the name(s), I must indicate this lack of information concisely'
    },
    {
        "task_id": "6f37996b-2ac7-44b0-8e68-6d28256631b4",
        "submitted_answer": 'b, e'
    },
    {
        "task_id": "9d191bce-651d-4746-be2d-7ef8ecadb9c2",
        "submitted_answer": 'Extremely'
    },
    {
        "task_id": "cabe07ed-9eca-40ea-8ead-410ef5e83f91",
        "submitted_answer": '• * The question asks "What is the surname...". Since the information is not available in the provided context, I cannot provide the surname. The final answer should be "as few words as possible". "Unknown" is a single word that accurately reflects the situation where the information cannot be retrieved from the given data. It is a string, uses no articles or abbreviations, and fits the requirements'
    },
    {
        "task_id": "3cef3a44-215e-4aed-8e3b-b1e3f08063b7",
        "submitted_answer": 'broccoli, celery, fresh basil, lettuce, sweet potatoes'
    },
    {
        "task_id": "99c9cc74-fdc8-46c6-8f8d-3ce2d3bfeea3",
        "submitted_answer": 'Cornstarch, Granulated sugar, Lemon juice, Ripe strawberries, Salt, Unsalted butter, Vanilla extract'
    },
    {
        "task_id": "305ac316-eef6-4446-960a-92d80d542f82",
        "submitted_answer": 'Wojciech'
    },
    {
        "task_id": "f918266a-b3e0-4914-865d-4faa564f1aef",
        "submitted_answer": 'Python code not provided'
    },
    {
        "task_id": "3f57289b-8c60-48be-bd80-01f8099ca449",
        "submitted_answer": '540'
    },
    {
        "task_id": "1f975693-876d-457b-a649-393859e79bf3",
        "submitted_answer": '15, 22, 23, 24, 25, 30, 41, 42, 43'
    },
    {
        "task_id": "840bfca7-4f7b-481a-8794-c560c340185d",
        "submitted_answer": '80GSFC21M0002'
    },
    {
        "task_id": "bda648d7-d618-4883-88f4-3466eabd860e",
        "submitted_answer": 'Saint Petersburg'
    },
    {
        "task_id": "cf106601-ab4f-4af9-b045-5295fe67b37d",
        "submitted_answer": 'CUB'
    },
    {
        "task_id": "a0c07678-e491-4bbc-8f0b-07405144218f",
        "submitted_answer": 'Nagai, VerHagen'
    },
    {
        "task_id": "7bd855d8-463d-4ed5-93ca-5fe35145f733",
        "submitted_answer": 'Excel file content needed'
    },
    {
        "task_id": "5a0c1adf-205e-4841-a666-7c3ef95def9d",
        "submitted_answer": 'Claus'
    },
]

def get_answer(task_id: str) -> dict:
    """Get answer for a specific task ID"""
    for answer in ANSWERS:
        if answer["task_id"] == task_id:
            return answer
    return {
        "task_id": task_id,
        "submitted_answer": "Task ID not found"
    }

def get_all_answers() -> list:
    """Get all answers"""
    return ANSWERS

def get_task_ids() -> list:
    """Get all task IDs"""
    return [answer["task_id"] for answer in ANSWERS]

def get_statistics() -> dict:
    """Get submission statistics"""
    total_tasks = len(ANSWERS)
    answered_tasks = sum(1 for answer in ANSWERS
                        if answer.get("submitted_answer", "").strip())
    
    return {
        "total_tasks": total_tasks,
        "answered_tasks": answered_tasks,
        "completion_rate": answered_tasks / total_tasks if total_tasks > 0 else 0.0
    }

if __name__ == "__main__":
    # Print statistics when run directly
    stats = get_statistics()
    print(f"GAIA Submission Statistics:")
    print(f"Total tasks: {stats['total_tasks']}")
    print(f"Answered tasks: {stats['answered_tasks']}")
    print(f"Completion rate: {stats['completion_rate']:.2%}")
