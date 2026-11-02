"""入夜派对：本地题库、场次限制与每题发送状态。"""

import re
import time
import unicodedata

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from utils import debug_exception, extract_ocr_text, logger
from utils.arena_state import BEIJING, now_beijing


# 题干用于核对关键词是否串题；整数答案表示从 1 开始逐个尝试的上限。
QUESTIONS = (
    ("13个人捉迷藏，捉了10个，还剩几个？", r"(?<!\d)13个.*捉.*(?<!\d)10个", "2"),
    ("1到10哪个不开车？", r"(?<!\d)1到10(?!\d).*不开车|一到十.*不开车", "10"),
    ("BJ4是什么意思？", "bj4", "不解释"),
    ("Kiss是动词、名词还是形容词？", "kiss", "连接词"),
    ("把8分成两半，是多少？", r"(?<!\d)8(?!\d).*两半|八.*两半", "0"),
    ("歌词填空：把你的心我的心，动作是什么？", "把你的[心新]", "串一串"),
    ("白猫掉进水里，黑猫救它上来，白猫说了什么？", "白猫", "喵"),
    ("爆竹声中一岁除，春风送暖入什么？", "春风送暖入(?!屠苏)", "屠苏"),
    ("爆竹声中一岁除春风送暖入屠苏，屠苏指的是什么？", "屠苏", "酒"),
    ("被你赞过的朋友圈叫什么圈？", "朋友圈", "甜甜圈"),
    ("本草纲目的作者是谁？", "本草纲目", "李时珍"),
    ("比火车还长的车是什么？", "火车", "塞车"),
    ("猜猜大喵今天吃了几块五花肉？", "几块五花肉", 9),
    ("猜猜大喵今天收到了几个红包？", "几个红包", 9),
    ("超人保护地球，谁保护城市？", "超人保护", "屏幕"),
    ("沉鱼落雁闭月羞花中的落雁是指谁？", "落雁", "昭君"),
    ("吃苹果时吃到几条虫子最可怕？", "虫子", "半条"),
    ("吃什么会变丑？", "变丑", "藕"),
    ("池小鱼的室友是？", "池小鱼", "一衣红雪"),
    ("大江东去，浪淘尽，千古风流什么？", "大江东去", "人物"),
    ("大喵的此刻ID是什么鉴赏大师？", "鉴赏大师", "五花肉"),
    ("大年三十又叫什么？", "大年三十", "除夕"),
    ("大熊猫一生的遗憾是没有什么？", "一生的遗憾", "彩色照片"),
    ("歌词填空：大雨让这座城市怎样变化？", "大雨.*城市", "颠倒"),
    ("第一次到暖暖家，她穿的套装是什么？", "第一次.*家", "欲望之音"),
    ("豆蔻年华是指女生多少岁？", "豆蔻年华", "13岁"),
    ("独立战争爆发在哪一个国家？", "独立战争", "苹果联邦"),
    ("歌词填空：耳边的话让人断肠也无怨，填的是什么？", "断肠", "爱我永不变"),
    ("浮梦立方是宙与海的什么？", "浮梦立方", "实验基地"),
    ("感人肺腑、赚人热泪的文章被形容加了什么？", "感人肺腑|赚人热泪", "洋葱"),
    ("葛朗台常被用来形容什么性格的人？", "葛朗台", "吝啬"),
    ("歌词填空：给妈妈做什么家务？", "给妈妈", "刷刷筷子洗洗碗"),
    ("古代的一个时辰在今天是多长时间？", "时辰", "两小时"),
    ("故年清梦的设计师是？", "故年清梦", "秦衣"),
    ("关羽字云长，刘备字玄德，伍佰字什么？", "伍佰", "心得"),
    ("《哈姆雷特》的作者是？", "哈姆雷特", "莎士比亚"),
    ("《海豹的自杀日记》是谁的作品？", "自杀日记", "茶薄荷"),
    ("海市蜃楼是光的折射还是反射造成的？", "海市蜃楼", "折射"),
    ("海哲创立的服装店全名是？", "海哲.*服装店", "苹果联邦服装集团"),
    ("汉语言文字的历史最早可以追溯到夏朝还是商朝？", "汉语言文字", "商朝"),
    ("好像棋盘似的，另一人说倒有点像什么？", "棋盘", "稿纸"),
    ("黑鹫就职于哪个财团？", "黑鹫", "墨丘利"),
    ("后羿射日传说中，射下的太阳是什么鸟？", "后羿", "乌鸦"),
    ("狐狸为什么站不起来？", "狐狸", "狡猾"),
    ("江北希在荒原拍摄的是？", "江北希", "国家地理"),
    ("金木水火土谁的腿最长？", "金木水火土", "火"),
    ("近朱者赤，近你者什么？", "近朱者赤", "甜"),
    ("开张大吉，猜一个地名。", "开张大吉", "新店"),
    ("歌词填空：可爱的男孩，你是什么样的女孩？", "可爱的男孩", "可爱"),
    ("来者何人？", "来者何人", "你的人"),
    ("狼去北极掉进冰河里，会变成什么？", "狼.*北极", "槟榔"),
    ("歌词填空：离开之后，风筝寻梦的地点？", "风筝", "天空"),
    ("里德断章是谁创作的？", "里德断章", "李尔里德"),
    ("两个小白叫什么？", "两个小白", "小白兔"),
    ("流影镇镇长的大名是？", "流影镇.*镇长", "罗德尔"),
    ("罗德尔是什么镇的镇长？", "罗德尔", "流影"),
    ("洛洛梨是哪个国家的优雅大使？", "优雅大使", "妮妮尔"),
    ("玛丽和梦露黏在一起变成什么？", "玛丽.*梦露", "玛丽莲梦露"),
    ("买橘子的片段是朱自清写给谁的文章？", "写给谁", "父亲"),
    ("没有邮筒，猜一成语。", "邮筒", "不可置信"),
    ("没有指向的地图，上北下南，左西右什么？", "左西右", "东"),
    ("美甲工作间每天可以从收银台收取什么？", "收银台", "心意币"),
    ("美甲工作间升级至几级，营业时将可能有特约顾客光临？", "特约顾客", "4"),
    ("蜜蜂停在日历上，猜一成语。", "蜜蜂", "风和日丽"),
    ("歌词填空：面对流言蜚语需要什么？", "流言蜚语", "勇气"),
    ("魔羊在废墟的编号是多少？", "魔羊", "219"),
    ("莫道不消魂，帘卷西风，人比黄花什么？", "人比黄花", "瘦"),
    ("墨丘利在此刻的VIP级别是？", "vip", "5"),
    ("哪个月有28天？", r"(?<!\d)28天", "每个月"),
    ("哪两个英文字母合在一起会爆炸？", "字母.*爆炸", "OK"),
    ("哪些英文字母是人最喜欢听的？", "字母.*喜欢听|喜欢听.*字母", "CD"),
    ("哪种动物的名字颠倒过来，会从动物变成食物？", "名字.*倒过来", "蜜蜂"),
    ("哪种动物最怕冷？", "动物最怕冷", "鸭子"),
    ("哪种水果视力最差？", "[视势]力最差", "芒果"),
    ("哪种竹子不长在土里？", "竹子", "爆竹"),
    ("你的脸上有点东西，有什么？", "脸上有点东西", "有点漂亮"),
    ("你是不是喜欢我，找出重复的字。", "你是不是喜欢我", "是"),
    ("歌词填空：你是唯一的什么？", "唯一的", "神话"),
    ("你属什么？", "你属", "属于我"),
    ("《你笑起来真好看》春天事物填空", "春天", "花儿"),
    ("年纪不大，胡子一把，只会叫妈妈，猜一种动物。", "胡子一.*把", "羊"),
    ("凝练馆是谁的秘密花园？", "秘密花园", "海"),
    ("牛顿因为什么发现万有引力？", "万有引力", "苹果"),
    ("暖暖吃的薯片是什么口味？", "薯片|洋芋片", "番茄"),
    ("暖暖出生在什么季节？", "出生.*季节", "冬季"),
    ("暖暖的什么东西经常掉下阳台？", "阳台", "手机"),
    ("暖暖看的电视剧中，女主角的恋人是爸爸的谁？", "恋人", "仇人的儿子"),
    ("暖暖是哪个星座的？", "星座", "射手座"),
    ("暖暖在足球校队中踢哪个位置？", "足球校队", "守门员"),
    ("暖暖最喜欢吃哪一种糖？", "吃哪一种糖", "话梅糖"),
    ("歌词填空：陪在一个陌生人左右，开头的时间是什么？", "陌生人", "十年之前"),
    ("苹果联邦服装集团的创始人是谁？", "服装集团的创始人", "海哲"),
    ("《七里香》电线杆上的麻雀在做什么？", "电线杆", "多嘴"),
    ("奇变偶不变，下一句是什么？", "奇变偶不变", "符号看象限"),
    ("奇迹大陆的社交软件叫什么？", "社交软件|通讯软件", "此刻"),
    ("奇迹大陆以科技闻名的国家是？", "科技闻名", "废墟孤岛"),
    ("麒麟到了北极会变成什么？", "麒麟", "冰淇淋"),
    ("汽车会飞，猜一种饮料？", "汽车会飞", "咖啡"),
    ("秦衣又被称为什么千颜？", "秦衣.*[千干](?:颜|面)", "卿"),
    ("《青春修炼手册》慢动作的后续填空", "慢动作", "重播"),
    ("清明时节雨什么，路上行人欲断魂？", "清明时节", "纷纷"),
    ("请选出图中服饰的正确名称（输入1、2、3、4进行回答）", "图中服饰", 4),
    ("屈原投身的河流叫什么？", "屈原", "汨罗江"),
    ("人有悲欢离合，月有什么？", "人有悲欢离合", "阴晴圆缺"),
    ("日心说是谁提出的？", "日心说", "哥白尼"),
    ("如果暖暖晚饭后就开始运动，她刚刚可能喝了什么？", "[晚晩]饭后", "奶茶"),
    ("如果有辆出租车，你是司机，小明是乘客，车是谁的？", "出租车|计程车", "如果"),
    ("三分熟牛排遇到五分熟牛排为什么没打招呼？", "牛排", "不熟"),
    ("三个小朋友猜拳，分别出剪刀石头布，共有多少手指？", r"(?:三个|(?<!\d)3个)小朋友.*(?:猜拳|剪刀|剪子|石头)", "30"),
    ("桑蚕丝制作的丝织品，应在多少摄氏度水温下单独手洗？", "丝织品", "40"),
    ("鲨鱼吃了一颗绿豆结果它变成了什么？", "鲨鱼", "绿豆沙"),
    ("山不在高，有仙则名，水不在什么，有龙则灵？", "有龙则灵", "深"),
    ("设计中心里暖暖的地图标示在谁的上方？", "设计中心", "薇薇安"),
    ("谁是万兽之王？", "万兽之王", "动物园园长"),
    ("谁天天晒网却不打鱼？", "晒网", "蜘蛛"),
    ("什么地方的门永远关不上？", "永远关不上", "球门"),
    ("什么东西以每小时2000公里载人，却不用燃料？", "2000公里|两千公里", "地球"),
    ("什么动物经常熬夜？", "熬夜", "熊猫"),
    ("什么动物可以贴在墙上？", "贴在墙上", "海豹"),
    ("什么动物最没有方向感？", "没有方向感", "麋鹿"),
    ("什么人不怕冷？", "人不怕冷", "雪人"),
    ("歌词填空：什么如此美妙，指引我们想要的未来？", "如此美妙", "爱是一道光"),
    ("什么生南国，春来发几枝？", "生南国", "红豆"),
    ("什么水果最忙？", "水果.*忙", "芒果"),
    ("什么熊最厉害？", "熊最厉害", "灰熊"),
    ("生存还是毁灭，是谁的一句名言？", "生存还是毁灭|tobeornottobe", "莎士比亚"),
    ("失败是成功之母，那成功是失败的什么？", "成功之母", "反义词"),
    ("师者，所以传道、什么、解惑也？", "师者", "授业"),
    ("时尚编辑室创刊号的名字是？", "创刊号", "甜心童话"),
    ("睡觉前绝对不会忘记的事情是什么？", "睡觉前", "闭眼"),
    ("四大菜系指的是山东菜四川菜广东菜和什么菜？", "四大菜系", "江苏"),
    ("台风天要带多少钱才能出门？", "台风|飓风", "40000000"),
    ("套装欲望魔术的帽子上的装饰是一只什么动物？", "欲望魔术", "兔子"),
    ("歌词填空：天黑风大，爸爸捕鱼，为什么还不做什么？", "爸爸捕鱼", "回家"),
    ("天增岁月人增寿，春满乾坤什么满门？", "春满乾坤", "福"),
    ("图鉴资料馆的全名是？", "图鉴资料", "忆海资料馆"),
    ("蚊子咬的一大一小两个包，大的是公蚊子还是母蚊子咬的？", "蚊子", "母蚊子"),
    ("我爸爸也有一双这样的袜子，指的是哪一双袜子？", "爸爸也有", "拼布菱格"),
    ("我买几个橘子，你就在此地不要动，这段话出自哪篇文章？", "橘子.*出自", "背影"),
    ("《无限少女》旅途填空：涂抹什么？", "涂抹", "色彩"),
    ("《五花肉之歌》是谁创作的？", "五花肉之歌", "大喵"),
    ("五月花和百合花谁没生过小孩？", "五月花", "五月花"),
    ("夏梦露设计的完美AI原型叫什么？", "夏梦露", "爱衣"),
    ("相对论是谁提出的？", "相对论", "爱因斯坦"),
    ("歌词填空：想象中有一双什么鞋？", "想[象像]中.*双", "滑板鞋"),
    ("歌词填空：小船儿推开波浪，前面的动作是什么？", "小船.*推开波浪", "荡起双桨"),
    ("小海最喜欢的设计师是？", "小海|阿海", "李尔里德"),
    ("小明的妈妈只生了三个孩子，老大大毛，老二二毛，老三叫什么？", "小明的?妈妈", "小明"),
    ("小王戴中分假发以后变成什么？", "中分假发", "小全"),
    ("小王子的星球上长了一株什么？", "小王子", "玫瑰"),
    ("歌词填空：小呀小苹果，下一句是什么？", "小[呀丫]小苹果", "怎么爱你都不嫌多"),
    ("星羽镇专属的童话是？", "星羽镇", "星羽天鹅"),
    ("形容一个人乐观，通常说他有什么精神？", "乐观", "阿Q"),
    ("薛定谔提出的一个实验叫薛定谔的什么？", "薛定谔", "猫"),
    ("歌词填空：颜色不一样的什么？", "颜色不一样", "烟火"),
    ("《洋葱》盘底洋葱的身份填空", "盘底", "调味品"),
    ("歌词填空：要感谢的人太多，不如谢什么？", "感谢的人", "天"),
    ("要红包时说什么发财，红包拿来？", "发财", "恭喜"),
    ("《野狼disco》右边画彩虹时，左边画什么？", "右边.*彩虹", "龙"),
    ("叶苏夏是苹果联邦流影镇几年级的小学生？", "几年级", "2"),
    ("叶苏夏最崇拜的人是谁？", "崇拜", "洛昂"),
    ("夜来风雨声，什么知多少？", "夜来风雨声", "花落"),
    ("夜宵和暖暖、大喵因为什么食物而认识？", "夜宵", "小笼包"),
    ("一般来说，失眠的人都在想什么？", "失眠", "睡觉"),
    ("歌词填空：一杯什么能换一夜不流泪？", "一夜不流泪", "忘情水"),
    ("一边是绿一边是红，一边喜雨一边喜风，猜一字。", "喜雨.*喜风", "秋"),
    ("歌词填空：一个人在人海做什么？", "在人海", "浮沉"),
    ("一颗好友的爱心可以获得多少体力？", "好友.*(?:爱心|体力)", "2"),
    ("一颗心值多少钱？", "一颗心", "一亿"),
    ("一口井7米深，蜗牛白天爬3米、晚上下坠2米，几天爬出？", "井7米.*蜗牛.*爬3米.*坠2米", "5"),
    ("一只鹿越跑越快最后变成了什么？", "鹿越跑越快", "高速公路"),
    ("一只小猪说加油啊，打一食品。", "小猪", "朱古力"),
    ("以铜为镜可以正衣冠，以史为镜可以知什么？", "以[史古]为镜", "兴替"),
    ("用椰子和西瓜打头，哪一个比较痛？", "椰子", "头"),
    ("由秦衣创立的剧团名称是？", "剧团名称", "幕不落剧团"),
    ("由薇薇安开设的设计师专栏名称是？", "设计师专栏", "设计师情报屋"),
    ("有一个字人人见了都会念错，是什么字？", "都会念错", "错"),
    ("佑果的室友是？", "佑果", "佐森"),
    ("鱼吐泡泡，猜一种颜色。", "鱼吐泡泡", "蓝色"),
    ("与暖暖的亲密度达到几级可以解锁设计工具？", "设计工具", "20"),
    ("玉米去烫了头变成了什么？", "玉米", "爆米花"),
    ("遇见你，我就姓什么？", "遇见你.*姓", "福"),
    ("元素顺序氢氦锂铍硼直到硫氯氩钾，下一项是什么？", "硫氯氩钾", "钙"),
    ("在哪里可以看到已拥有的设计师之影？", "已拥有.*设计师之影", "方舟"),
    ("在奇迹大陆中哪个国家覆雪面积最大？", "覆(?:盖)?雪面积", "北地王国"),
    ("在树上唱rap，猜一个字。", "树上.*rap", "桑"),
    ("在云端帝国中最具代表性的食物是什么？", "代表性的食物", "火锅"),
    ("珍珠不宜密封保存，需要与什么接触？", "珍珠", "空气"),
    ("正月里来是新年，大年什么头一天？", "头一天", "初一"),
    ("只骗中年人，猜一成语。", "中年人", "童叟无欺"),
    ("钻石的计量单位是什么？", "计量单位", "克拉"),
    ("最坚不可摧的门怕什么？", "坚不可摧", "钥匙"),
    ("左一组建的雇佣兵团叫什么？", "雇佣兵团", "审判者军团"),
    ("歌词填空：坐在谷堆旁，听谁讲故事？", "谷堆", "妈妈"),
)


def normalize(text):
    return re.sub(r"[^\w]", "", unicodedata.normalize("NFKC", text).casefold())


def match_question(text):
    text = normalize(text)
    matches = [i for i, (_, pattern, _) in enumerate(QUESTIONS) if re.search(pattern, text)]
    return matches[0] if len(matches) == 1 else None


def session_end(now=None):
    now = (now or now_beijing()).astimezone(BEIJING)
    if now.hour in (10, 19, 20) and 10 <= now.minute < 20:
        return now.replace(minute=20, second=0, microsecond=0)
    return None


def panel_state(text):
    text = normalize(text)
    if "本次答题结束" in text:
        return "ended"
    if "秒后进入" in text:
        return "waiting"
    return "question"


@AgentServer.custom_action("union_party")
class UnionPartyAction(CustomAction):
    def run(self, context, argv):
        try:
            now = now_beijing()
            end = session_end(now)
            if context.tasker.stopping:
                return False
            if end is None:
                logger.info("入夜派对尚未开放")
                return True
            deadline = time.monotonic() + (end - now).total_seconds() + 30
            return PartyRunner(context, deadline).run()
        except Exception as exc:
            debug_exception("入夜派对执行失败")
            if not context.tasker.stopping:
                logger.error(f"入夜派对：{exc}" if isinstance(exc, RuntimeError)
                             else "入夜派对：执行异常")
            return False


class PartyRunner:
    def __init__(self, context, deadline):
        self.context = context
        self.deadline = deadline

    def check(self):
        if self.context.tasker.stopping:
            raise RuntimeError("任务已停止")
        if time.monotonic() >= self.deadline:
            raise RuntimeError("未识别到答题结束提示")

    def capture(self):
        self.check()
        controller = self.context.tasker.controller
        if not controller.post_screencap().wait().succeeded or controller.cached_image is None:
            raise RuntimeError("截图失败")
        self.check()
        return controller.cached_image

    def recognize(self, node, image):
        result = self.context.run_recognition(node, image)
        if result is None:
            raise RuntimeError("识别执行失败")
        self.check()
        return result

    def read_panel(self, image):
        result = self.recognize("入夜派对读取题目", image)
        if not result.hit:
            return ""
        # 整块题目可能分成多行；共用的 extract_ocr_text 只取首条结果。
        items = result.all_results or result.filtered_results
        if not items:
            return extract_ocr_text(result)
        return "\n".join(item.text for item in sorted(items, key=lambda item: (item.box[1], item.box[0])))

    def action(self, node, box=(0, 0, 0, 0)):
        self.check()
        result = self.context.run_action(node, box=box)
        if result is None or not result.success:
            raise RuntimeError("点击失败")
        self.check()

    def task(self, node, override=None):
        self.check()
        result = self.context.run_task(node, pipeline_override=override)
        if result is None or not result.status.succeeded:
            raise RuntimeError("文本输入失败" if node == "入夜派对输入答案" else "进入入夜派对失败")
        self.check()

    def wait_button(self, node, question=None):
        deadline = min(self.deadline, time.monotonic() + 30)
        while True:
            image = self.capture()
            if question is not None:
                text = self.read_panel(image)
                other = match_question(text)
                if panel_state(text) != "question" or (other is not None and other != question):
                    return None
                if time.monotonic() >= self.deadline - 30:
                    return None
            result = self.recognize(node, image)
            if result.hit and result.box[2] > 0 and result.box[3] > 0:
                return result.box
            if time.monotonic() >= deadline:
                raise RuntimeError("未找到入夜派对入口" if question is None else "未找到输入或发送按钮")
            time.sleep(.2)

    def send(self, question, answer):
        box = self.wait_button("入夜派对输入框", question)
        if box is None:
            return False
        self.action("入夜派对输入框", box)
        self.task("入夜派对输入答案", {"入夜派对输入答案": {"action": {
            "type": "Custom", "param": {"custom_action": "dreamweaver_input_text",
                "custom_action_param": {"input_text": answer}},
        }}})
        self.action("入夜派对点击空白")
        box = self.wait_button("入夜派对发送", question)
        if box is None:
            return False
        self.action("入夜派对发送", box)
        # 按约定发送后固定点击两次，不检测奖励弹窗。
        for _ in range(2):
            self.action("入夜派对点击空白")
        logger.debug(f"入夜派对：题目={QUESTIONS[question][0]}，发送={answer}")
        logger.info("入夜派对：已发送答案")
        return True

    def enter(self):
        if self.recognize("导航在入夜派对页面", self.capture()).hit:
            return True
        self.task("入夜派对任务链准备")
        box = self.wait_button("入夜派对点击前往派对")
        self.action("入夜派对点击前往派对", box)
        # 点击后立即截图，先检查短暂提示；不重试点击，也不依赖提示中的标点。
        deadline = min(self.deadline, time.monotonic() + 30)
        while True:
            image = self.capture()
            if self.recognize("入夜派对今日已参与", image).hit:
                logger.info("入夜派对：今日已参与，跳过")
                return False
            if self.recognize("导航在入夜派对页面", image).hit:
                return True
            if time.monotonic() >= deadline:
                raise RuntimeError("未能进入答题页面")
            time.sleep(.1)

    def run(self):
        if not self.enter():
            return True
        logger.info("入夜派对：开始答题")
        current, sent, waiting, unknown_logged = None, 0, False, False
        while True:
            text = self.read_panel(self.capture())
            state = panel_state(text)
            if state == "ended":
                logger.info("入夜派对：本次答题已结束")
                return True
            if state == "waiting":
                waiting, unknown_logged = True, False
            else:
                question = match_question(text)
                if question is None:
                    if normalize(text) and not unknown_logged:
                        logger.debug(f"入夜派对：未匹配题目={text!r}")
                        logger.info("入夜派对：未找到答案，等待下一题")
                        unknown_logged = True
                else:
                    if waiting or question != current:
                        current, sent, waiting, unknown_logged = question, 0, False, False
                    answer = QUESTIONS[question][2]
                    limit = answer if isinstance(answer, int) else 1
                    if sent < limit and time.monotonic() < self.deadline - 30:
                        if self.send(question, str(sent + 1) if isinstance(answer, int) else answer):
                            sent += 1
            # 空白或 OCR 漏字不清除 current/sent，只有换题才允许重新发送。
            time.sleep(.3)
