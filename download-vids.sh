vids=(
    "fall_01_falling_off_bike__4kU-hbhQ9is_000000_000010.mp4"
    "fall_02_falling_off_bike__7qZvo1A3MRQ_000000_000010.mp4"
    "fall_03_falling_off_bike__QXlLnjld17U_000014_000024.mp4"
    "fall_04_falling_off_bike__SocGMDzqthw_000000_000010.mp4"
    "fall_05_falling_off_chair__-5hw88bD4mE_000003_000013.mp4"
    "fall_06_falling_off_chair__-tTBQvmKfEs_000002_000012.mp4"
    "fall_07_falling_off_chair__kwh65tVZyik_000001_000011.mp4"
    "fall_08_face_planting__2zHDlGlWTG8_000003_000013.mp4"
    "fall_09_face_planting__DnxygkTssDc_000004_000014.mp4"
    "fall_10_face_planting__JR9LTG9F4j0_000003_000013.mp4"
    "general_01_battle_rope_training__-i2ZylN33X0_000098_000108.mp4"
    "general_02_playing_squash_or_racquetball__5foqSJcNZtU_000012_000022.mp4"
    "general_03_building_lego__-nQAhz9mI_0_000026_000036.mp4"
    "general_04_planting_trees__GMEfgxoCdTo_000039_000049.mp4"
    "general_05_brush_painting__5kJpNKicmBc_000269_000279.mp4"
    "general_06_hurdling__37ETnFtq_Hk_000000_000010.mp4"
    "general_07_calligraphy__89hqPkPYWUs_000124_000134.mp4"
    "general_08_bowling__2K8Zo7PVcr4_000001_000011.mp4"
    "general_09_playing_flute__6HyNydVIji4_000207_000217.mp4"
    "general_10_laying_concrete__1MmOQj66w94_000197_000207.mp4"
)
names=(
    "fall01.mp4" "fall02.mp4" "fall03.mp4" "fall04.mp4" "fall05.mp4"
    "fall06.mp4" "fall07.mp4" "fall08.mp4" "fall09.mp4" "fall10.mp4"
    "general01.mp4" "general02.mp4" "general03.mp4" "general04.mp4" "general05.mp4"
    "general06.mp4" "general07.mp4" "general08.mp4" "general09.mp4" "general10.mp4"
)

for i in "${!vids[@]}"; do
    vid="${vids[$i]}"
    name="${names[$i]}"
    echo "Downloading $vid as vids/$name"
    curl -o vids/$name -L https://github.com/bear7066/gemma4-action-demo/raw/refs/heads/main/videos/$vid
done

