import asyncio
from backend.services.video_search import search_videos, inject_video_citations

async def main():
    # 测试 Bilibili 搜索
    videos = await search_videos("快速排序算法")
    print(f"搜索到 {len(videos)} 个视频:")
    for v in videos:
        print(f"  - {v.title} ({v.duration}) {v.url}")

    # 测试引用注入
    draft = "快速排序是一种分治排序算法。\n\n它的平均时间复杂度为 O(n log n)。"
    result = inject_video_citations(draft, videos)
    print("\n--- 注入后的文档 ---")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())