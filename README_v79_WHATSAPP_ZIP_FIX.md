# Production Test Dashboard v79

## Direct WhatsApp ZIP parser

v79 fixes the false error:

> No usable production-test data was found in the ZIP archive.

The failure could occur when a WhatsApp export contained only `_chat.txt` plus
unsupported media such as an `.opus` voice note. The v78 facade recursively sent
`_chat.txt` through the generic file loader and silently ignored exceptions. If
that route returned no table, the complete ZIP was reported as empty even though
the chat contained valid PICO/TMU readings.

## Changes

- Parses `_chat.txt` directly before processing attachments.
- Supports Android bracket exports and iOS/plain WhatsApp timestamp formats.
- Handles UTF-8 BOM, UTF-16, Arabic Windows encoding and safe fallbacks.
- Tries both the mature WhatsApp-export parser and the robust TMU block parser.
- Chooses the interpretation with the strongest unique timestamped readings.
- Removes repeated quoted/edited duplicates.
- Ignores audio/video/sticker files without treating the ZIP as invalid.
- Continues parsing Excel, CSV, PDF, Word and optional image attachments.
- Provides useful ZIP diagnostics instead of swallowing every member error.
- Changes the parser build ID so Streamlit discards the old cached failure.

## Validation with the supplied field export

`WhatsApp Chat - Sitra8-58 clean & test(2).zip` contains `_chat.txt` and one
`.opus` voice note. With image processing disabled, v79 returns:

- 74 unique production-test readings
- Well: S8-58
- Start: 12-Jun-2026 00:00
- End: 16-Jun-2026 15:00
- Source: WhatsApp export text
- Duplicate well/timestamp rows: 0

The voice note is ignored because it is not required for production-test plots.
