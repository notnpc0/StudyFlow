import math
import json
import os
from datetime import date, datetime, timedelta


# ============================================================
# FILE / DATABASE SETTINGS
# ============================================================

DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "study_data.json")


# ============================================================
# ADAPTIVE PLANNER TUNING CONSTANTS
# ============================================================

# Controls how quickly "mastery pressure" (the need-to-study pressure
# from low mastery) falls off as mastery approaches 100%. A value of 1
# is a straight line (the old behaviour): pressure crashes to near-zero
# as soon as your marks look good. A value below 1 bows that line
# upward, so a topic sitting at 90-95% mastery still carries real
# pressure - closing the last few percent is genuinely harder than the
# first 50%. At exactly 100% mastery, pressure is always 0.
MASTERY_PRESSURE_EXPONENT = 0.5

# How many days before an exam the planner should stop splitting time
# across every topic and instead dedicate the whole day to whichever
# topic's exam is that close.
EXAM_EVE_DAYS_BEFORE = 1


# ============================================================
# DATE FUNCTIONS
# ============================================================

def get_date(text):
    return datetime.strptime(
        text,
        "%Y-%m-%d"
    ).date()


def days_until(exam_date):
    return max(
        0,
        (exam_date - date.today()).days
    )


# ============================================================
# DATA STORAGE
# ============================================================

def save_data(data):

    with open(DATA_FILE, "w") as file:
        json.dump(
            data,
            file,
            indent=4
        )


def load_data():

    if not os.path.exists(DATA_FILE):
        return None

    with open(DATA_FILE, "r") as file:
        return json.load(file)


# ============================================================
# MEMORY MODEL
# ============================================================

def calculate_memory(
    mastery,
    days_since_study,
    forgetting_rate
):

    memory = (
        mastery
        * math.exp(
            -forgetting_rate
            * days_since_study
        )
    )

    return max(
        0,
        min(1, memory)
    )


# ============================================================
# URGENCY
# ============================================================

def calculate_urgency(days_left):

    if days_left <= 0:
        return 1

    return 1 / days_left


# ============================================================
# ADAPTIVE PID GAINS
# ============================================================

def calculate_adaptive_gains(
    performance,
    previous_performance,
    persistent_error
):

    # ----------------------------
    # Kp
    # Lower performance = higher Kp
    # ----------------------------

    Kp_min = 0.2
    Kp_max = 1.0

    Kp = (
        Kp_min
        + (Kp_max - Kp_min)
        * (1 - performance)
    )

    # ----------------------------
    # Ki
    # Persistent underperformance
    # ----------------------------

    Ki_min = 0.02
    Ki_max = 0.10

    error_normalized = min(
        1,
        persistent_error
    )

    Ki = (
        Ki_min
        + (Ki_max - Ki_min)
        * error_normalized
    )

    # ----------------------------
    # Kd
    # Magnitude of performance change
    # ----------------------------

    Kd_min = 0.05
    Kd_max = 0.20

    performance_change = abs(
        performance
        - previous_performance
    )

    change_normalized = min(
        1,
        performance_change
    )

    Kd = (
        Kd_min
        + (Kd_max - Kd_min)
        * change_normalized
    )

    return Kp, Ki, Kd


# ============================================================
# PID CONTROLLER
# ============================================================

def calculate_pid(
    Kp,
    Ki,
    Kd,
    error,
    previous_error,
    error_sum,
    delta_t
):

    proportional = Kp * error

    integral = (
        Ki
        * error_sum
        * delta_t
    )

    if delta_t == 0:
        derivative = 0
    else:
        derivative = (
            Kd
            * (error - previous_error)
            / delta_t
        )

    adjustment = (
        proportional
        + integral
        + derivative
    )

    return adjustment


# ============================================================
# PRIORITY CALCULATION
# ============================================================

def calculate_priority(topic):

    # Curved instead of linear: at 90% mastery this works out to about
    # 0.316 rather than 0.1, and at 95% about 0.224 rather than 0.05,
    # so a topic that's nearly mastered still carries real pressure
    # instead of priority collapsing to near-zero as soon as marks
    # look good.

    mastery_pressure = (
        (1 - topic["mastery"])
        ** MASTERY_PRESSURE_EXPONENT
    )

    memory_pressure = (
        1 - topic["memory"]
    )

    urgency = topic["urgency"]

    difficulty_pressure = (
        topic["difficulty"] / 10
    )

    remaining_work = (
        topic["required_minutes"]
        - topic["completed_minutes"]
    )

    if remaining_work < 0:
        remaining_work = 0

    work_pressure = min(
        1,
        remaining_work / max(
            1,
            topic["required_minutes"]
        )
    )

    priority = (
        mastery_pressure
        + memory_pressure
        + urgency
        + difficulty_pressure
        + work_pressure
    )

    return priority


# ============================================================
# UPDATE TOPIC
# ============================================================

def update_topic(topic):

    today = date.today()

    last_studied = get_date(
        topic["last_studied"]
    )

    days_since = (
        today - last_studied
    ).days

    days_left = (
        get_date(topic["exam_date"])
        - today
    ).days

    # NOTE: days_left is intentionally left unclamped (it can go
    # negative once an exam has passed). build_schedule() and
    # calculate_workload() both rely on being able to tell a topic
    # whose exam already happened apart from one that's still
    # upcoming, so they can exclude it from future planning.

    memory = calculate_memory(
        topic["mastery"],
        days_since,
        topic["forgetting_rate"]
    )

    urgency = calculate_urgency(
        days_left
    )

    topic["days_since_study"] = days_since

    topic["days_until_exam"] = days_left

    topic["memory"] = memory

    topic["urgency"] = urgency

    topic["priority"] = calculate_priority(
        topic
    )

    return topic


# ============================================================
# PERSONALIZED FORGETTING RATE
# ============================================================

def update_forgetting_rate(topic):

    tests = topic["tests"]

    if len(tests) < 2:
        return

    estimates = []

    mastery = max(
        0.01,
        topic["mastery"]
    )

    for test in tests:

        days_elapsed = test[
            "days_after_study"
        ]

        score = test[
            "percentage"
        ]

        if days_elapsed > 0 and score > 0:

            retention = (
                score / 100
            )

            try:

                rate = (
                    -math.log(
                        retention
                        / mastery
                    )
                    / days_elapsed
                )

                if rate > 0:
                    estimates.append(
                        rate
                    )

            except ValueError:
                pass

    if estimates:

        new_rate = (
            sum(estimates)
            / len(estimates)
        )

        # Prevent unrealistic values

        topic["forgetting_rate"] = max(
            0.001,
            min(
                1,
                new_rate
            )
        )


# ============================================================
# CREATE TOPIC
# ============================================================

def create_topic(
    subject_name,
    exam_date
):

    name = input(
        "Topic name: "
    )

    difficulty = float(
        input(
            "Difficulty (1-10): "
        )
    )

    required_minutes = float(
        input(
            "Estimated study time needed (minutes): "
        )
    )

    marks = input(
        "Current topic score (e.g. 17/20): "
    )

    current, maximum = marks.split("/")

    current = float(current)
    maximum = float(maximum)

    mastery = current / maximum

    topic = {

        "name": name,

        "subject": subject_name,

        "exam_date": str(
            exam_date
        ),

        "difficulty": difficulty,

        "required_minutes":
            required_minutes,

        "completed_minutes": 0,

        "current_marks": current,

        "maximum_marks": maximum,

        "mastery": mastery,

        "memory": mastery,

        "urgency": 0,

        "priority": 0,

        "forgetting_rate": 0.05,

        "last_studied": str(
            date.today()
        ),

        "days_since_study": 0,

        "days_until_exam": 0,

        "tests": [],

        "study_history": []
    }

    update_topic(topic)

    return topic


# ============================================================
# CREATE SUBJECT
# ============================================================

def create_subject():

    name = input(
        "\nSubject name: "
    ).strip()

    marks = input(
        "Current subject score (e.g. 87/100): "
    )

    current, maximum = marks.split("/")

    current = float(current)
    maximum = float(maximum)

    exam_text = input(
        "Exam date (YYYY-MM-DD): "
    )

    exam_date = get_date(
        exam_text
    )

    subject = {

        "name": name,

        "current_marks": current,

        "maximum_marks": maximum,

        "exam_date": str(
            exam_date
        ),

        "topics": []
    }

    print(
        "\nNow enter the topics."
    )

    topic_count = int(
        input(
            "How many topics? "
        )
    )

    for _ in range(topic_count):

        topic = create_topic(
            name,
            exam_date
        )

        subject["topics"].append(
            topic
        )

    return subject


# ============================================================
# TOTAL TOPIC INFORMATION
# ============================================================

def get_all_topics(data):

    topics = []

    for subject in data["subjects"]:

        for topic in subject["topics"]:

            topics.append(topic)

    return topics


# ============================================================
# UPDATE ALL TOPICS
# ============================================================

def update_all_topics(data):

    for topic in get_all_topics(data):

        update_forgetting_rate(
            topic
        )

        update_topic(
            topic
        )


# ============================================================
# STUDY CAPACITY
# ============================================================

def calculate_capacity(
    start_date,
    end_date,
    weekday_minutes,
    weekend_minutes
):

    current = start_date

    total = 0

    while current <= end_date:

        if current.weekday() < 5:
            total += weekday_minutes
        else:
            total += weekend_minutes

        current += timedelta(
            days=1
        )

    return total


# ============================================================
# WORKLOAD ANALYSIS
# ============================================================

def calculate_workload(data):

    topics = get_all_topics(data)

    # Exams that have already passed no longer count toward required
    # work - there's nothing left to prepare for.

    active_topics = [
        topic
        for topic in topics
        if topic["days_until_exam"] >= 0
    ]

    required = 0
    completed = 0

    for topic in active_topics:

        required += (
            topic["required_minutes"]
        )

        completed += (
            topic["completed_minutes"]
        )

    remaining = max(
        0,
        required - completed
    )

    if not active_topics:

        return (
            required,
            completed,
            remaining,
            0,
            0
        )

    latest_exam = max(
        get_date(topic["exam_date"])
        for topic in active_topics
    )

    capacity = calculate_capacity(
        date.today(),
        latest_exam,
        data["weekday_minutes"],
        data["weekend_minutes"]
    )

    if capacity == 0:
        workload_ratio = float("inf")
    else:
        workload_ratio = (
            remaining / capacity
        )

    return (
        required,
        completed,
        remaining,
        capacity,
        workload_ratio
    )


# ============================================================
# ROTATION
# ============================================================

def create_rotation(
    topics,
    subjects_per_day
):

    ranked = sorted(
        topics,
        key=lambda topic:
            topic["priority"],
        reverse=True
    )

    rotation = []

    for topic in ranked:

        # High-priority topics appear
        # more often in the rotation.

        appearances = max(
            1,
            math.ceil(
                topic["priority"]
            )
        )

        for _ in range(
            appearances
        ):

            rotation.append(
                topic
            )

    return rotation


# ============================================================
# BUILD SCHEDULE
# ============================================================

def build_schedule(
    data,
    days
):

    topics = get_all_topics(data)

    topics = [
        topic
        for topic in topics
        if topic["days_until_exam"] >= 0
    ]

    rotation = create_rotation(
        topics,
        data["subjects_per_day"]
    )

    schedule = []

    position = 0

    for day_number in range(days):

        current_date = (
            date.today()
            + timedelta(
                days=day_number
            )
        )

        if current_date.weekday() < 5:
            available = (
                data["weekday_minutes"]
            )
        else:
            available = (
                data["weekend_minutes"]
            )

        # Exam-eve exclusivity: if any topic's exam falls exactly
        # EXAM_EVE_DAYS_BEFORE after this scheduled day, that day is
        # dedicated entirely to it (or to all of them, if more than
        # one topic shares that exam date) instead of splitting focus
        # across unrelated topics.

        exam_eve_topics = [
            topic
            for topic in topics
            if (
                get_date(topic["exam_date"])
                - current_date
            ).days == EXAM_EVE_DAYS_BEFORE
        ]

        if exam_eve_topics:

            daily_topics = exam_eve_topics

        else:

            daily_topics = []

            attempts = 0

            while (
                len(daily_topics)
                < data["subjects_per_day"]
                and attempts < len(rotation) * 2
            ):

                if len(rotation) == 0:
                    break

                topic = rotation[
                    position % len(rotation)
                ]

                position += 1

                attempts += 1

                if (
                    topic["name"]
                    not in [
                        t["name"]
                        for t in daily_topics
                    ]
                ):

                    # Don't schedule a topic
                    # after its exam.

                    exam_date = get_date(
                        topic["exam_date"]
                    )

                    if current_date <= exam_date:

                        daily_topics.append(
                            topic
                        )

        schedule.append(
            {
                "date": str(
                    current_date
                ),

                "available_minutes":
                    available,

                "topics":
                    [
                        topic["name"]
                        for topic
                        in daily_topics
                    ]
            }
        )

    return schedule


# ============================================================
# DAILY TIME ALLOCATION
# ============================================================

def allocate_daily_time(
    topic_names,
    data,
    available_minutes
):

    topics = get_all_topics(data)

    selected = []

    for topic in topics:

        if topic["name"] in topic_names:

            selected.append(topic)

    total_priority = sum(
        topic["priority"]
        for topic in selected
    )

    allocations = {}

    if total_priority == 0:

        return allocations

    for topic in selected:

        percentage = (
            topic["priority"]
            / total_priority
        )

        allocations[
            topic["name"]
        ] = (
            available_minutes
            * percentage
        )

    return allocations


# ============================================================
# RECORD STUDY SESSION
# ============================================================

def record_study_session(
    topic,
    planned_minutes,
    actual_minutes=None
):

    print(
        "\nRecording:",
        topic["name"]
    )

    if actual_minutes is None:
        actual = float(
            input(
                "Minutes actually studied: "
            )
        )
    else:
        actual = float(actual_minutes)

    actual = max(
        0,
        actual
    )

    session = {

        "date":
            str(date.today()),

        "planned_minutes":
            planned_minutes,

        "actual_minutes":
            actual
    }

    topic["study_history"].append(
        session
    )

    topic["completed_minutes"] += (
        actual
    )

    topic["last_studied"] = str(
        date.today()
    )

    topic["days_since_study"] = 0

    return actual


# ============================================================
# RECORD TEST
# ============================================================

def record_test(
    topic,
    test_topic=None,
    current=None,
    maximum=None
):

    print(
        "\nTest for:",
        topic["name"]
    )

    if test_topic is None:
        test_topic = input(
            "Specific topic tested: "
        )

    if current is None or maximum is None:
        marks = input(
            "Test marks (e.g. 18/20): "
        )
        current, maximum = marks.split("/")

    current = float(current)
    maximum = float(maximum)

    percentage = (
        current
        / maximum
    ) * 100

    last_studied = get_date(
        topic["last_studied"]
    )

    elapsed = (
        date.today()
        - last_studied
    ).days

    test = {

        "topic":
            test_topic,

        "date":
            str(date.today()),

        "marks":
            current,

        "maximum_marks":
            maximum,

        "percentage":
            percentage,

        "days_after_study":
            elapsed
    }

    topic["tests"].append(
        test
    )

    # Update mastery.

    topic["mastery"] = (
        percentage / 100
    )

    topic["current_marks"] = current

    topic["maximum_marks"] = maximum

    # Learning from the test

    update_forgetting_rate(
        topic
    )

    update_topic(
        topic
    )


# ============================================================
# ADAPTIVE CONTROLLER
# ============================================================

def run_controller(data):

    topics = get_all_topics(data)

    if len(topics) == 0:
        return

    # Average current performance

    performance = sum(
        topic["mastery"]
        for topic in topics
    ) / len(topics)

    previous_performance = data.get(
        "previous_performance",
        performance
    )

    # Persistent error

    errors = data.get(
        "performance_errors",
        []
    )

    current_error = max(
        0,
        1 - performance
    )

    errors.append(
        current_error
    )

    # Keep recent history only

    errors = errors[-10:]

    persistent_error = min(
        1,
        sum(errors)
    )

    Kp, Ki, Kd = (
        calculate_adaptive_gains(
            performance,
            previous_performance,
            persistent_error
        )
    )

    error = (
        1 - performance
    )

    previous_error = (
        1 - previous_performance
    )

    error_sum = sum(errors)

    adjustment = calculate_pid(
        Kp,
        Ki,
        Kd,
        error,
        previous_error,
        error_sum,
        1
    )

    data["Kp"] = Kp
    data["Ki"] = Ki
    data["Kd"] = Kd

    data["pid_adjustment"] = (
        adjustment
    )

    data["previous_performance"] = (
        performance
    )

    data["performance_errors"] = (
        errors
    )


# ============================================================
# DISPLAY DASHBOARD
# ============================================================

def display_dashboard(data):

    update_all_topics(data)

    print(
        "\n=========================================="
    )

    print(
        "          ADAPTIVE STUDY PLANNER"
    )

    print(
        "=========================================="
    )

    print(
        "Today:",
        date.today()
    )

    print(
        "Weekday capacity:",
        data["weekday_minutes"],
        "min"
    )

    print(
        "Weekend capacity:",
        data["weekend_minutes"],
        "min"
    )

    print(
        "\nSUBJECTS"
    )

    for subject in data["subjects"]:

        print(
            "\n",
            subject["name"],
            "| Exam:",
            subject["exam_date"]
        )

        for topic in subject["topics"]:

            status = (
                " (EXAM PASSED)"
                if topic["days_until_exam"] < 0
                else ""
            )

            print(
                "  ",
                topic["name"] + status,
                "| Mastery:",
                round(
                    topic["mastery"] * 100,
                    1
                ),
                "%",
                "| Priority:",
                round(
                    topic["priority"],
                    3
                )
            )


# ============================================================
# DISPLAY WORKLOAD
# ============================================================

def display_workload(data):

    (
        required,
        completed,
        remaining,
        capacity,
        ratio
    ) = calculate_workload(data)

    print(
        "\n=========================================="
    )

    print(
        "             WORKLOAD ANALYSIS"
    )

    print(
        "=========================================="
    )

    print(
        "Total required:",
        round(required, 1),
        "minutes"
    )

    print(
        "Completed:",
        round(completed, 1),
        "minutes"
    )

    print(
        "Remaining:",
        round(remaining, 1),
        "minutes"
    )

    print(
        "Available capacity:",
        round(capacity, 1),
        "minutes"
    )

    if ratio == float("inf"):

        print(
            "Workload ratio: INFINITE"
        )

    else:

        print(
            "Workload / capacity:",
            round(
                ratio * 100,
                1
            ),
            "%"
        )

    if ratio > 1:

        print(
            "\nWARNING:"
        )

        print(
            "Required work is greater than"
        )

        print(
            "your available study capacity."
        )

    else:

        print(
            "\nWorkload currently fits"
        )

        print(
            "inside available capacity."
        )


# ============================================================
# DISPLAY PID
# ============================================================

def display_controller(data):

    print(
        "\n=========================================="
    )

    print(
        "          ADAPTIVE CONTROLLER"
    )

    print(
        "=========================================="
    )

    print(
        "Kp:",
        round(
            data.get("Kp", 0),
            4
        )
    )

    print(
        "Ki:",
        round(
            data.get("Ki", 0),
            4
        )
    )

    print(
        "Kd:",
        round(
            data.get("Kd", 0),
            4
        )
    )

    print(
        "PID adjustment:",
        round(
            data.get(
                "pid_adjustment",
                0
            ),
            3
        )
    )


# ============================================================
# DISPLAY SCHEDULE
# ============================================================

def display_schedule(
    schedule,
    data
):

    print(
        "\n=========================================="
    )

    print(
        "            ROTATING SCHEDULE"
    )

    print(
        "=========================================="
    )

    for day in schedule:

        print(
            "\n",
            day["date"],
            "|",
            day["available_minutes"],
            "minutes"
        )

        print(
            "Topics:",
            ", ".join(
                day["topics"]
            )
        )

    # Today's detailed allocation

    if schedule:

        today = schedule[0]

        allocations = (
            allocate_daily_time(
                today["topics"],
                data,
                today["available_minutes"]
            )
        )

        print(
            "\nTODAY'S TIME ALLOCATION"
        )

        for name, minutes in (
            allocations.items()
        ):

            print(
                name,
                "→",
                round(
                    minutes,
                    1
                ),
                "minutes"
            )


# ============================================================

# ============================================================
# STUDYFLOW WEB SERVER
# ============================================================
#
# This file is both:
#   1. The adaptive study-planning engine
#   2. The Flask backend for index.html
#
# Only these files are required:
#   main.py
#   index.html
#   style.css
#
# study_data.json is created automatically.
# ============================================================

from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__, static_folder=".", static_url_path="")


# ============================================================
# HELPER FUNCTIONS FOR THE WEBSITE
# ============================================================

def find_topic(data, topic_name):
    """Find a topic by name."""
    for topic in get_all_topics(data):
        if topic.get("name") == topic_name:
            return topic
    return None


def save_and_update(data, run_pid=False):
    """Refresh topic calculations and save the database."""
    update_all_topics(data)

    if run_pid:
        run_controller(data)

    save_data(data)


def make_initial_data(
    weekday_minutes,
    weekend_minutes,
    subjects_per_day,
    subjects
):
    """Create the same data structure used by the original planner."""

    data = {
        "weekday_minutes": float(weekday_minutes),
        "weekend_minutes": float(weekend_minutes),
        "subjects_per_day": int(subjects_per_day),
        "subjects": subjects,
        "previous_performance": 0,
        "performance_errors": [],
        "Kp": 0,
        "Ki": 0,
        "Kd": 0,
        "pid_adjustment": 0
    }

    update_all_topics(data)
    run_controller(data)
    save_data(data)

    return data


def clean_topic(topic, subject_name, exam_date):
    """
    Convert a topic received from the browser into the format
    expected by the adaptive engine.
    """

    # Accept several useful browser field names.
    name = str(
        topic.get("name", topic.get("topic", "Untitled topic"))
    ).strip()

    try:
        difficulty = float(topic.get("difficulty", 5))
    except (TypeError, ValueError):
        difficulty = 5

    difficulty = max(1, min(10, difficulty))

    try:
        required_minutes = float(
            topic.get(
                "required_minutes",
                topic.get("minutes", 60)
            )
        )
    except (TypeError, ValueError):
        required_minutes = 60

    required_minutes = max(0, required_minutes)

    # Allow either mastery as a decimal or a mark such as 17/20.
    mastery = topic.get("mastery")

    if mastery is None:
        current_marks = topic.get(
            "current_marks",
            topic.get("current", 0)
        )
        maximum_marks = topic.get(
            "maximum_marks",
            topic.get("maximum", 100)
        )

        try:
            current_marks = float(current_marks)
            maximum_marks = float(maximum_marks)
            mastery = (
                current_marks / maximum_marks
                if maximum_marks > 0
                else 0
            )
        except (TypeError, ValueError, ZeroDivisionError):
            current_marks = 0
            maximum_marks = 100
            mastery = 0
    else:
        try:
            mastery = float(mastery)

            # If browser sends 85 instead of 0.85, convert it.
            if mastery > 1:
                mastery /= 100

            mastery = max(0, min(1, mastery))

            maximum_marks = float(
                topic.get("maximum_marks", 100)
            )
            current_marks = mastery * maximum_marks

        except (TypeError, ValueError):
            mastery = 0
            current_marks = 0
            maximum_marks = 100

    return {
        "name": name,
        "subject": subject_name,
        "exam_date": str(exam_date),
        "difficulty": difficulty,
        "required_minutes": required_minutes,
        "completed_minutes": float(
            topic.get("completed_minutes", 0)
        ),
        "current_marks": current_marks,
        "maximum_marks": maximum_marks,
        "mastery": mastery,
        "memory": mastery,
        "urgency": 0,
        "priority": 0,
        "forgetting_rate": float(
            topic.get("forgetting_rate", 0.05)
        ),
        "last_studied": topic.get(
            "last_studied",
            str(date.today())
        ),
        "days_since_study": 0,
        "days_until_exam": 0,
        "tests": topic.get("tests", []),
        "study_history": topic.get("study_history", [])
    }


def clean_subject(subject):
    """Convert browser subject data into the planner format."""

    name = str(subject.get("name", "Unnamed subject")).strip()

    exam_date = subject.get("exam_date")

    if not exam_date:
        exam_date = str(
            date.today() + timedelta(days=30)
        )

    # Validate the date so bad browser input does not crash the server.
    try:
        exam_date = get_date(str(exam_date))
    except (TypeError, ValueError):
        exam_date = date.today() + timedelta(days=30)

    topics = []

    for raw_topic in subject.get("topics", []):
        topics.append(
            clean_topic(
                raw_topic,
                name,
                exam_date
            )
        )

    return {
        "name": name,
        "current_marks": float(
            subject.get("current_marks", 0)
        ),
        "maximum_marks": float(
            subject.get("maximum_marks", 100)
        ),
        "exam_date": str(exam_date),
        "topics": topics
    }


# ============================================================
# WEBSITE
# ============================================================

@app.route("/")
def home():
    return send_from_directory(
        os.path.dirname(os.path.abspath(__file__)),
        "index.html"
    )


@app.route("/style.css")
def css():
    return send_from_directory(
        os.path.dirname(os.path.abspath(__file__)),
        "style.css"
    )


# ============================================================
# DATA
# ============================================================

@app.route("/api/data", methods=["GET"])
def api_data():

    data = load_data()

    if data is None:
        return jsonify({
            "setup_required": True,
            "subjects": []
        })

    # Refresh calculated values, but DO NOT run the PID controller
    # on every browser refresh. run_controller() records a new error
    # each time it is called, so repeatedly opening the dashboard
    # should not count as another performance observation.
    update_all_topics(data)
    save_data(data)

    return jsonify(data)


# ============================================================
# INITIAL SETUP
# ============================================================

@app.route("/api/setup", methods=["POST"])
def api_setup():

    body = request.get_json(silent=True) or {}

    try:
        weekday_minutes = float(
            body.get("weekday_minutes", 120)
        )

        weekend_minutes = float(
            body.get("weekend_minutes", 120)
        )

        subjects_per_day = int(
            body.get("subjects_per_day", 2)
        )

    except (TypeError, ValueError):
        return jsonify({
            "error": "Invalid study-time or subjects-per-day values."
        }), 400

    if weekday_minutes < 0 or weekend_minutes < 0:
        return jsonify({
            "error": "Study time cannot be negative."
        }), 400

    if subjects_per_day < 1:
        return jsonify({
            "error": "You need at least one subject/topic per day."
        }), 400

    raw_subjects = body.get("subjects", [])

    if not isinstance(raw_subjects, list):
        return jsonify({
            "error": "subjects must be a list."
        }), 400

    subjects = [
        clean_subject(subject)
        for subject in raw_subjects
        if isinstance(subject, dict)
    ]

    if not subjects:
        return jsonify({
            "error": "Add at least one subject."
        }), 400

    data = make_initial_data(
        weekday_minutes,
        weekend_minutes,
        subjects_per_day,
        subjects
    )

    return jsonify({
        "success": True,
        "data": data
    })


# ============================================================
# TOPICS
# ============================================================

@app.route("/api/topics", methods=["GET"])
def api_topics():

    data = load_data()

    if data is None:
        return jsonify([])

    update_all_topics(data)
    save_data(data)

    return jsonify(
        get_all_topics(data)
    )


# ============================================================
# SCHEDULE
# ============================================================

@app.route("/api/schedule", methods=["POST"])
def api_schedule():

    data = load_data()

    if data is None:
        return jsonify({
            "error": "Create your study plan first."
        }), 400

    body = request.get_json(silent=True) or {}

    try:
        days = int(body.get("days", 7))
    except (TypeError, ValueError):
        days = 7

    days = max(1, min(days, 30))

    update_all_topics(data)

    schedule = build_schedule(
        data,
        days
    )

    # Add the actual time allocation to each day.
    for day in schedule:
        day["allocations"] = allocate_daily_time(
            day["topics"],
            data,
            day["available_minutes"]
        )

    save_data(data)

    return jsonify(schedule)


# ============================================================
# RECORD STUDY SESSION
# ============================================================

@app.route("/api/study", methods=["POST"])
def api_study():

    data = load_data()

    if data is None:
        return jsonify({
            "error": "Create your study plan first."
        }), 400

    body = request.get_json(silent=True) or {}

    topic_name = body.get("topic")

    if not topic_name:
        return jsonify({
            "error": "Topic is required."
        }), 400

    try:
        actual_minutes = float(
            body.get("actual_minutes", body.get("minutes", 0))
        )

        planned_minutes = float(
            body.get("planned_minutes", 0)
        )

    except (TypeError, ValueError):
        return jsonify({
            "error": "Study minutes must be numbers."
        }), 400

    if actual_minutes < 0:
        return jsonify({
            "error": "Study time cannot be negative."
        }), 400

    topic = find_topic(
        data,
        topic_name
    )

    if topic is None:
        return jsonify({
            "error": "Topic not found."
        }), 404

    record_study_session(
        topic,
        planned_minutes,
        actual_minutes
    )

    save_and_update(
        data,
        run_pid=False
    )

    return jsonify({
        "success": True,
        "topic": topic,
        "data": data
    })


# ============================================================
# RECORD TEST
# ============================================================

@app.route("/api/test", methods=["POST"])
def api_test():

    data = load_data()

    if data is None:
        return jsonify({
            "error": "Create your study plan first."
        }), 400

    body = request.get_json(silent=True) or {}

    topic_name = body.get("topic")
    test_topic = body.get(
        "test_topic",
        body.get("tested_topic", topic_name)
    )

    mark = body.get(
        "mark",
        body.get("current")
    )

    maximum = body.get(
        "maximum",
        body.get("maximum_marks")
    )

    if not topic_name:
        return jsonify({
            "error": "Topic is required."
        }), 400

    if mark is None or maximum is None:
        return jsonify({
            "error": "Mark and maximum mark are required."
        }), 400

    try:
        mark = float(mark)
        maximum = float(maximum)
    except (TypeError, ValueError):
        return jsonify({
            "error": "Mark and maximum mark must be numbers."
        }), 400

    if maximum <= 0:
        return jsonify({
            "error": "Maximum mark must be greater than zero."
        }), 400

    if mark < 0 or mark > maximum:
        return jsonify({
            "error": "Mark must be between 0 and the maximum mark."
        }), 400

    topic = find_topic(
        data,
        topic_name
    )

    if topic is None:
        return jsonify({
            "error": "Topic not found."
        }), 404

    # Use the existing engine's test-recording logic.
    record_test(
        topic,
        test_topic=str(test_topic),
        current=mark,
        maximum=maximum
    )

    # The test is a new performance observation, so now the PID
    # controller is allowed to react to it.
    save_and_update(
        data,
        run_pid=True
    )

    return jsonify({
        "success": True,
        "topic": topic,
        "controller": {
            "Kp": data.get("Kp", 0),
            "Ki": data.get("Ki", 0),
            "Kd": data.get("Kd", 0),
            "pid_adjustment": data.get(
                "pid_adjustment",
                0
            )
        },
        "data": data
    })


# ============================================================
# CONTROLLER INFORMATION
# ============================================================

@app.route("/api/controller", methods=["GET"])
def api_controller():

    data = load_data()

    if data is None:
        return jsonify({
            "error": "Create your study plan first."
        }), 400

    return jsonify({
        "Kp": data.get("Kp", 0),
        "Ki": data.get("Ki", 0),
        "Kd": data.get("Kd", 0),
        "pid_adjustment": data.get(
            "pid_adjustment",
            0
        ),
        "previous_performance": data.get(
            "previous_performance",
            0
        ),
        "performance_errors": data.get(
            "performance_errors",
            []
        )
    })


# ============================================================
# WORKLOAD
# ============================================================

@app.route("/api/workload", methods=["GET"])
def api_workload():

    data = load_data()

    if data is None:
        return jsonify({
            "error": "Create your study plan first."
        }), 400

    update_all_topics(data)

    (
        required,
        completed,
        remaining,
        capacity,
        ratio
    ) = calculate_workload(data)

    return jsonify({
        "required": required,
        "completed": completed,
        "remaining": remaining,
        "capacity": capacity,
        "ratio": ratio
    })


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/api/health", methods=["GET"])
def api_health():

    return jsonify({
        "status": "ok",
        "planner": "StudyFlow",
        "engine": "active"
    })


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    print()
    print("==========================================")
    print("          STUDYFLOW")
    print("==========================================")
    print()
    print("Server running at:")
    print("http://127.0.0.1:5000")
    print()
    print("Keep this terminal open while using StudyFlow.")
    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
