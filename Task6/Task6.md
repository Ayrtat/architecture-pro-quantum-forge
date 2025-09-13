# Автоматическое ежедневное обновление базы знаний

1. Был создан скрипт [update_index.py](./update_index.py) работающий примерно по той же схеме, что и build_index.py
2. Запуск осуществляется по `cron`:
- `crontab -e`
- `0 6 * * * /usr/bin/python3 /path/to/update_index.py >> /tmp/cron.log 2>&1`
3. Диаграмма обновления: [diagram](./diagrams/update.puml)

Пример лога:


```
2025-09-13 16:13:12,162 - INFO - Initializing Knowledge Base Update
2025-09-13 16:13:15,707 - INFO - Load pretrained SentenceTransformer: all-mpnet-base-v2
2025-09-13 16:13:19,542 - INFO - Anonymized telemetry enabled. See                     https://docs.trychroma.com/telemetry for more information.
2025-09-13 16:13:19,678 - INFO - Retrieved: 32 documents
2025-09-13 16:13:19,678 - INFO - No changes detected in documents
2025-09-13 16:13:19,774 - INFO - 
Query: Who is Lord of Air?
2025-09-13 16:13:19,850 - INFO -   1. AlAkir.txt
2025-09-13 16:13:19,850 - INFO -      Zyphor

Zyphor the Windruler was the supernatural being King of air. He ruled over his domain of Skywall from his seat at the Throne of the Four Winds...
2025-09-13 16:13:19,850 - INFO -   2. Thunderaan.txt
2025-09-13 16:13:19,850 - INFO -      Stormrider the Windruler is the supernatural being King of air and master of Skywall. Formerly known as the Prince of Air, Stormrider became the new W...
2025-09-13 16:13:19,851 - INFO -   3. Therazane.txt
2025-09-13 16:13:19,851 - INFO -      Terralith the Stonemother is the supernatural being King of earth, and formerly one of the Old Almightys' lieutenants. As the ruler of Deepholm and mo...
2025-09-13 16:13:19,851 - INFO - 
Query: Who is enraged Fire Elemental?
2025-09-13 16:13:19,911 - INFO -   1. Smolderon.txt
2025-09-13 16:13:19,911 - INFO -      Fireruler Cindrath is the supernatural being King of fire and master of the Firelands. He had been fighting for power against Pyroth following the dea...
2025-09-13 16:13:19,911 - INFO -   2. Ragnaros.txt
2025-09-13 16:13:19,911 - INFO -      Flamorth the Fireruler was the incredibly powerful supernatural being King of Fire. After being summoned to Elysium Realm by Sorcerer-Thane Thaurissan...
2025-09-13 16:13:19,911 - INFO -   3. Ragnaros.txt
2025-09-13 16:13:19,911 - INFO -      The Wildhammer and Bronzebeard armies united to destroy the treacherous Dark Irons once and for all and marched for Thaurissan. Thaurissan, seeking to...
2025-09-13 16:13:19,911 - INFO - 
Query: Who is fallen titan who turned to demon?
2025-09-13 16:13:19,974 - INFO -   1. Sargeras.txt
2025-09-13 16:13:19,975 - INFO -      Nexthar (pronounced "SAHR-gair-ahs")[5] is a titan and the creator and leader of the Burning Legion. He was once the champion of the Pantheon, chosen ...
2025-09-13 16:13:19,975 - INFO -   2. Illidan.txt
2025-09-13 16:13:19,975 - INFO -      Zylarin Stormrage, commonly known as the Betrayer, is the first of the demon hunters, the former self-proclaimed King of Outland, the former ruler of ...
2025-09-13 16:13:19,975 - INFO -   3. Archimonde.txt
2025-09-13 16:13:19,975 - INFO -      Over time, however, a demonic presence began to infiltrate Gorathul's mind, using Thal'kiel's adorned skull as a conduit. When he slept, Gorathul saw ...
2025-09-13 16:13:19,976 - INFO - 
==================================================
2025-09-13 16:13:19,976 - INFO - Update Complete:
2025-09-13 16:13:19,976 - INFO -   Model: all-mpnet-base-v2
2025-09-13 16:13:19,976 - INFO -   Embedding Dimension: 768
2025-09-13 16:13:19,976 - INFO -   Documents Processed: 32
2025-09-13 16:13:19,976 - INFO -   Chunks Added: 104
2025-09-13 16:13:19,976 - INFO -   Total Chunks: 104
2025-09-13 16:13:19,976 - INFO -   Processing Time: 0.13 minutes
2025-09-13 16:13:19,976 - INFO - ==================================================s
```