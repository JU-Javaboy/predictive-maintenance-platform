# Chinedu Godwin Nwobodo — Project Journal

## Week of 16 November 2026

- Set up the project on my laptop with uv and Python 3.13.
- Explored the 3-year dataset. It has 525,600 rows, 18 columns and 20 machines.
- Found that many lines in the CSV had extra quotes around them. This broke normal CSV reading.
- Found 1,076 missing temperature values and no duplicate rows.
- Found some odd values, like negative probabilities and very high temperature spikes.
- Wrote our data cleaning and validation plan with the team.
- Agreed on how to handle missing data:
  - Less than 5% missing: fill in the gaps.
  - 5% to 10% missing: drop those rows.
  - More than 10% missing: reject the whole batch.
- Wrote the C1 cleaning code:
  - parser.py reads the raw CSV and fixes the quoted lines.
  - schemas.py checks that each column has the right type.
  - quality.py has all the cleaning rules.
- Cleaning rules I added:
  - Check timestamps, machine IDs and operating modes.
  - Find duplicate rows.
  - Set aside values that are physically impossible as per domain knowledge.
  - Flag sensor spikes instead of deleting them.
- Filled missing sensor values with the median for each operating mode. I used the median instead of the average because a few very high spikes (temperature goes up to 1,091 °C) would pull the average up. I worked out the medians from the first two years of data only, so the last year stays unseen for testing.
- Put all the numbers we may want to change in one settings file so it's easier for the team to maintain.
- Made sure bad data does not crash the program.
- Small batches (under 100 rows) are not rejected as a whole. We only drop the bad rows.
- Added two columns to the cleaned file to show what we did to each row: quality_disposition and quality_flags.
- Wrote a script that cleans the whole 3-year file using the same rules.
- Ran the script and further explored the cleaned file. No missing values left.
- Wrote 31 tests for the cleaning rules. All of them pass.
- Refactored the code.
- Planned how the API will receive new data batches. We will save a raw copy and a cleaned copy of every batch.
