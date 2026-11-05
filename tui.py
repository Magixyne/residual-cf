"""Fast offline full-screen terminal UI for cached MovieLens recommendations."""
from __future__ import annotations

from math import ceil

from prompt_toolkit.application import Application
from prompt_toolkit.application.current import get_app_or_none
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, VSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.styles import Style
from prompt_toolkit.utils import get_cwidth
from prompt_toolkit.widgets import Frame, TextArea

PAGE_SIZE = 8

STYLE = Style.from_dict({
    "header": "bold #f2f6fa bg:#24496e",
    "title": "bold #24496e",
    "muted": "#627284",
    "accent": "bold #2e6b58",
    "score": "bold #9a6225",
    "error": "bold #a43838",
    "table-head": "bold #f4f7fa bg:#41566b",
    "help": "#eaf0f6 bg:#344a60",
})


def _cell(value: str, width: int) -> str:
    """Clip/pad a terminal cell using display width rather than Python len()."""
    value = str(value)
    clipped = ""
    used = 0
    truncated = False
    for char in value:
        char_width = get_cwidth(char)
        if used + char_width > width:
            truncated = True
            break
        clipped += char
        used += char_width
    if truncated and width > 0:
        while clipped and used + 1 > width:
            last = clipped[-1]
            clipped = clipped[:-1]
            used -= get_cwidth(last)
        clipped += "…"
        used += 1
    return clipped + " " * max(0, width - used)


def run_tui(data, initial_user=66):
    users = data["users"]
    state = {"user_id": int(initial_user), "error": "", "page": 0}
    app = None

    def invalidate():
        state["page"] = 0
        if app is not None:
            app.invalidate()

    def select_all(buffer):
        buffer.cursor_position = 0
        buffer.start_selection()
        buffer.cursor_position = len(buffer.text)

    def accept_user(buffer):
        if not buffer.text.strip():
            if app is not None:
                app.invalidate()
            return
        try:
            user_id = int(buffer.text.strip())
            if str(user_id) not in users:
                raise ValueError
            state["user_id"] = user_id
            state["error"] = ""
            state["page"] = 0
            buffer.text = ""
            buffer.cursor_position = 0
        except ValueError:
            state["error"] = "用户 ID 不存在；请检查后按 Enter。"
            select_all(buffer)
        if app is not None:
            app.invalidate()

    user_input = TextArea(
        text="", height=1, multiline=False,
        accept_handler=accept_user, style="class:input", name="user_id",
    )
    genre_input = TextArea(
        text="", height=1, multiline=False, style="class:input", name="genre",
    )
    search_input = TextArea(
        text="", height=1, multiline=False, style="class:input", name="title",
    )
    genre_input.buffer.on_text_changed += lambda _: invalidate()
    search_input.buffer.on_text_changed += lambda _: invalidate()

    def matches(movie, genre, query):
        return (
            (not genre or genre in movie["genres"].casefold())
            and (not query or query in movie["title"].casefold())
        )

    def render():
        fragments = []

        def add(style, text):
            fragments.append((style, text))

        ds = data["dataset"]
        params = data["params"]
        user_params = params["user_cf"]
        item_params = params["item_cf"]
        uid = str(state["user_id"])
        user = users.get(uid)
        add("class:header", "  MOVIELENS  /  HYBRID COLLABORATIVE FILTERING\n")
        add("class:title", "  电影推荐系统\n")
        add(
            "class:muted",
            f"  {ds['ratings']:,} 条评分  |  {ds['users']:,} 位用户  |  "
            f"{ds['rated_movies']:,} 部有评分电影\n\n",
        )
        if state["error"]:
            add("class:error", f"  {state['error']}\n\n")
        if user is None:
            add("class:error", "  找不到该用户 ID。\n")
            return fragments

        add("class:title", f"  用户 {uid}  ")
        add("class:muted", f"{user['rated_count']} 条评分  |  平均分 {user['mean_rating']:.2f}\n")
        app = get_app_or_none()
        term_width = app.output.get_size().columns if app else 100
        liked = "  /  ".join(
            f"{movie['title']} ({movie['rating']:.1f})" for movie in user["top_liked"][:2]
        )
        if len(user["top_liked"]) > 2:
            liked += "  /  …"
        liked_width = max(8, term_width - 14)
        add("class:muted", f"  高分电影：{_cell(liked or '暂无', liked_width)}\n\n")

        genre = genre_input.text.strip().casefold()
        query = search_input.text.strip().casefold()
        candidates = [
            movie for movie in user["recommendations"]
            if matches(movie, genre, query)
        ]
        page_count = max(1, ceil(len(candidates) / PAGE_SIZE))
        state["page"] = min(state["page"], page_count - 1)
        start = state["page"] * PAGE_SIZE
        page_items = candidates[start:start + PAGE_SIZE]

        table_width = max(54, term_width - 4)
        fixed_width = 27
        title_width = max(22, int((table_width - fixed_width) * 0.63))
        genre_width = max(12, table_width - fixed_width - title_width)
        header = (
            "  #  " + _cell("电影", title_width) + " "
            + _cell("类型", genre_width) + "  预测分   支持权重\n"
        )
        add("class:table-head", header)
        add("class:muted", "  " + "─" * table_width + "\n")
        if not candidates:
            add("class:muted", "  当前筛选下没有结果；列表覆盖模型前 50 条候选。\n")
        for index, movie in enumerate(page_items, start=start + 1):
            title = _cell(movie["title"], title_width)
            genres = _cell(movie["genres"].replace("|", ", "), genre_width)
            add("class:muted", f"  {index:02d}  {title} {genres}  ")
            add("class:score", f"{movie['predicted_rating']:>5.2f}  ")
            add("class:accent", f"{movie['neighbor_weight']:>8.3f}\n")

        shown_start = start + 1 if candidates else 0
        shown_end = start + len(page_items)
        add("\n", "\n")
        add("class:muted", f"  筛选结果 {len(candidates)} 条  |  当前 {shown_start}–{shown_end} / {len(candidates)}\n")
        test_metrics = data.get("metrics", {}).get("加法：偏差+UserCF+ItemCF")
        metric_text = (
            f" RMSE={test_metrics['RMSE']:.4f} MAE={test_metrics['MAE']:.4f}"
            if test_metrics else ""
        )
        add(
            "class:accent",
            f"  UserCF={user_params['k_neighbors']}  ItemCF={item_params['k_neighbors']}  "
            f"混合={params['hybrid_scale']:g}{metric_text}\n",
        )
        return fragments

    body = Window(
        content=FormattedTextControl(render),
        wrap_lines=False,
        always_hide_cursor=True,
    )
    controls = VSplit([
        Frame(user_input, title="用户 ID · Enter", width=22),
        Frame(genre_input, title="类型过滤", width=25),
        Frame(search_input, title="片名搜索"),
    ], height=3)
    footer = Window(
        content=FormattedTextControl([(
            "class:help",
            "  F2/F3 翻页   Tab 切换输入框   Ctrl-U 清空   Ctrl-Q 退出   类型示例：Drama / Comedy  ",
        )]),
        height=1,
    )
    root = HSplit([controls, body, footer])
    bindings = KeyBindings()

    @bindings.add("c-q")
    @bindings.add("c-c")
    def _(event):
        event.app.exit()

    @bindings.add("tab")
    def _(event):
        event.app.layout.focus_next()

    @bindings.add("s-tab")
    def _(event):
        event.app.layout.focus_previous()

    @bindings.add("f2")
    def _(event):
        state["page"] = max(0, state["page"] - 1)
        event.app.invalidate()

    @bindings.add("f3")
    def _(event):
        user = users.get(str(state["user_id"]), {})
        genre = genre_input.text.strip().casefold()
        query = search_input.text.strip().casefold()
        count = sum(
            1 for movie in user.get("recommendations", [])
            if matches(movie, genre, query)
        )
        state["page"] = min(max(0, ceil(count / PAGE_SIZE) - 1), state["page"] + 1)
        event.app.invalidate()

    app = Application(
        layout=Layout(root, focused_element=user_input),
        key_bindings=bindings,
        style=STYLE,
        full_screen=True,
        mouse_support=True,
    )
    app.run()


def print_user(data, user_id):
    user = data["users"].get(str(user_id))
    if user is None:
        raise ValueError(f"User ID {user_id} is not present in demo_data.json")
    print(f"User {user_id} | {user['rated_count']} ratings | mean {user['mean_rating']:.2f}")
    print(f"{'#':>2}  {'Title':<52} {'Genres':<34} {'Score':>5} {'Support':>8}")
    print("-" * 108)
    for index, movie in enumerate(user["recommendations"], start=1):
        print(
            f"{index:>2}  {movie['title'][:52]:<52} "
            f"{movie['genres'].replace('|', ', ')[:34]:<34} "
            f"{movie['predicted_rating']:>5.2f} {movie['neighbor_weight']:>8.3f}"
        )
