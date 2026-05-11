# Redrob Resume Matching Engine

Resume Matching Engine built for the Redrob AI Campus Hackathon.

## Problem Statement

Build a system that:

- Normalizes noisy resume skill data
- Applies alias mapping
- Deduplicates canonical skills
- Builds TF-IDF vectors for resumes
- Builds binary vectors for Job Descriptions (JDs)
- Computes cosine similarity
- Ranks the Top 3 matching candidates for each JD

## Features

- Skill normalization using provided `SKILL_ALIASES`
- Multi-word phrase matching
- Deduplication of skills
- Manual TF-IDF implementation
- Binary JD vector creation
- Cosine similarity ranking
- No external libraries used

## Tech Stack

- Python
- Standard Library (`math`)

## Project Structure

```text
resume_matching.py
README.md
