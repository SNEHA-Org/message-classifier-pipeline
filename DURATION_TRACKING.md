# Duration Tracking Feature

## Overview

The audio transcription script now tracks and displays the duration of the entire process, including:
- ⏱️ **Per-batch duration** - How long each batch takes to process
- ⏱️ **Total duration** - Complete time from start to finish
- 📊 **Average time per audio** - Average processing time per audio file

## Output Format

### During Processing

Each batch shows its processing time:
```
✅ Batch 1 complete: 50 transcribed, 0 failed. (took 2m 5s)
✅ Batch 2 complete: 48 transcribed, 2 failed. (took 1m 58s)
```

### Final Summary

At the end, you'll see comprehensive duration statistics:
```
================== 📊 SUMMARY ==================
Total rows fetched:       100
Successfully transcribed: 98
Failed transcriptions:    2
Total batches processed:  2
Total duration:           4m 5s
Average time per audio:   2.50s
================================================
```

## Duration Formatting

Durations are automatically formatted in human-readable format:
- **Seconds only**: `45s`
- **Minutes and seconds**: `2m 30s`
- **Hours, minutes, seconds**: `1h 15m 30s`
- **Days, hours, minutes, seconds**: `2d 3h 45m 15s`

## Metrics Explanation

### Per-Batch Duration
- **What it measures**: Time from fetching the batch to updating the database
- **Includes**:
  - Database query time
  - Audio download time
  - Whisper transcription time (parallel processing)
  - GPT-4 transliteration time
  - Database update time
- **Excludes**: Sleep time between batches

### Total Duration
- **What it measures**: Complete wall-clock time from script start to finish
- **Includes**:
  - All batch processing time
  - Sleep time between batches
  - Database connection setup
  - Any overhead

### Average Time Per Audio
- **What it measures**: Mean processing time per successfully transcribed audio
- **Calculation**: `Total duration / Successfully transcribed count`
- **Use case**: Estimate time needed for future batches
- **Note**: Only shown if at least one audio was successfully processed

## Performance Benchmarks

Typical processing times (will vary based on):
- Audio file length
- Network speed
- OpenAI API response time
- Database latency
- Number of parallel workers

**Example estimates**:
- Short audio (10-30 seconds): ~2-5 seconds per file
- Medium audio (30-60 seconds): ~5-10 seconds per file
- Long audio (60+ seconds): ~10-20 seconds per file

**Factors affecting speed**:
- `MAX_WORKERS` setting (default: 5 parallel threads)
- Audio file size and format
- Network bandwidth
- OpenAI API rate limits
- Database connection speed

## Using Duration Data for Planning

### Estimate Processing Time

If you have 1,000 audio files to process and average time is 3 seconds per audio:
```
Estimated time = (1000 / 50) * (3 * 50 + 1) = 20 batches * 151 seconds ≈ 50 minutes
```

Where:
- 50 = `BATCH_SIZE`
- 3 = average seconds per audio
- 1 = `SLEEP_BETWEEN_BATCHES` in seconds

### Optimization Tips

**To improve speed**:
1. **Increase parallel workers**:
   ```python
   MAX_WORKERS = 10  # Process more files simultaneously
   ```
   ⚠️ Be careful not to exceed OpenAI API rate limits

2. **Increase batch size**:
   ```python
   BATCH_SIZE = 100  # Process more files per batch
   ```

3. **Reduce sleep time** (if database can handle it):
   ```python
   SLEEP_BETWEEN_BATCHES = 0.5  # Shorter pause between batches
   ```

4. **Use faster network connection** for audio downloads

5. **Optimize database queries** (already using SQLAlchemy with batching)

## Logging

Duration information is displayed in the console but not logged to the error log file. To save duration data:

```bash
# Save output to a log file
python3 audio_transcription.py | tee processing_log_$(date +%Y%m%d_%H%M%S).txt
```

## Example Sessions

### Small Run (No Data)
```
🚀 Starting multilingual transcription process...
🔄 Fetching a batch...
📥 Rows fetched: 0

================== 📊 SUMMARY ==================
Total rows fetched:       0
Successfully transcribed: 0
Failed transcriptions:    0
Total batches processed:  0
Total duration:           2s
================================================
```

### Medium Run (200 audio files)
```
🚀 Starting multilingual transcription process...
🔄 Fetching a batch...
📥 Rows fetched: 50
✅ Batch 1 complete: 50 transcribed, 0 failed. (took 2m 15s)

🔄 Fetching a batch...
📥 Rows fetched: 50
✅ Batch 2 complete: 48 transcribed, 2 failed. (took 2m 8s)

🔄 Fetching a batch...
📥 Rows fetched: 50
✅ Batch 3 complete: 50 transcribed, 0 failed. (took 2m 12s)

🔄 Fetching a batch...
📥 Rows fetched: 50
✅ Batch 4 complete: 49 transcribed, 1 failed. (took 2m 10s)

🔄 Fetching a batch...
📥 Rows fetched: 0

================== 📊 SUMMARY ==================
Total rows fetched:       200
Successfully transcribed: 197
Failed transcriptions:    3
Total batches processed:  4
Total duration:           8m 48s
Average time per audio:   2.68s
================================================
```

## Code Details

The duration tracking is implemented using:
- `time.time()` for precise timestamps
- `datetime.timedelta` for human-readable formatting
- Per-batch timing with `batch_start_time`
- Global timing with `start_time`

All timing code has minimal performance impact (<1ms overhead).

