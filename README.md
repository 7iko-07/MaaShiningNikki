# MaaShiningNikki

**闪现吧暖暖**，一个基于 [MaaFramework](https://github.com/MaaXYZ/MaaFramework) 的 **闪耀暖暖小助手**。  
通过图像识别、OCR 与模拟控制，把日常重复操作交给自动化流程处理。

如果 MaaShiningNikki 对你有帮助，欢迎在项目右上角点亮 Star 支持。

</div>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white">
  <img alt="MaaFramework" src="https://img.shields.io/badge/MaaFramework-5.x-blue">
  <img alt="platform" src="https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-blueviolet">
  <br>
  <img alt="license" src="https://img.shields.io/github/license/7iko-07/MaaShiningNikki">
  <img alt="commit" src="https://img.shields.io/github/commit-activity/m/7iko-07/MaaShiningNikki">
  <img alt="stars" src="https://img.shields.io/github/stars/7iko-07/MaaShiningNikki?style=social">
  <img alt="downloads" src="https://img.shields.io/github/downloads/7iko-07/MaaShiningNikki/total?style=social">

<br>

[![qq group](https://img.shields.io/badge/QQ%E7%BE%A4-977642793-hotpink)](https://qm.qq.com/q/NQA8iFWmQg)

</p>

## 功能概览

当前项目采用 **Maa Pipeline JSON + Python Agent 自定义动作** 的混合架构。常规点击、识别和跳转由 Pipeline 描述，战力比较、动态翻页、商品判断、列表匹配等复杂逻辑由 Python Agent 处理。

| 模块 | 当前支持 |
| --- | --- |
| 启动与导航 | 登录页点击开始、通用加载等待、返回主页、页面导航 |
| 好友 | 好友一键送心、联盟一键送心、体力一键领取 |
| 联盟 | 机密任务执行、联盟金币捐献、联盟福利领取 |
| 邮件与福利 | 邮件奖励领取、每日签到、免费体力补给 |
| 结伴 | 时光钟表铺、不落的帷幕、印象旅航 |
| 美甲 | 雇佣免费店员、提取心意币、特约顾客、美甲点赞 |
| 制作目标服装 | 获取制衣引导目标服装所需的时空回廊材料 |
| 情报屋 | 自动调查左一情报 |
| 回家 | 采购番茄炒蛋、采购金币小物、自习室好感度获取 |
| 竞技场 | 识别双方战力、刷新弱对手、剩余次数循环挑战、可选搭配服装 |
| 搭配评选赛 | 自动点赞 |
| 一键领取 | 活跃任务、时尚任务、时尚提升计划奖励领取 |
| 活动 | 活动剧情/活动任务流程，当前包含“织吻为拥” |
| 回忆小铺 | 识别六个商品，支持金币商品处理、钻石商品折扣/价格阈值暂停、免费刷新判断 |

> [!NOTE]  
> 项目目前主要围绕安卓端和 mumu模拟器开发。其他分辨率、模拟器或系统若出现识别偏移，请优先附带[日志](https://github.com/SuperWaterGod/MaaGakumasu/blob/main/docs/zh_cn/%E6%8F%90%E9%97%AE%E4%B8%8E%E5%8F%8D%E9%A6%88%E6%8C%87%E5%8D%97.md#%E5%AF%BC%E5%87%BA%E8%B0%83%E8%AF%95%E6%97%A5%E5%BF%97)

## 注意事项

1. 推荐使用720p以上并保持游戏画面完整显示。
6. 本项目仅用于学习交流，请勿用于商业用途。
7. 本项目仅提供自动化脚本，不提供任何游戏资源。

## 使用说明

### Windows

在 [Releases](https://github.com/7iko-07/MaaShiningNikki/releases) 下载对应版本压缩包：

| 架构 | 下载文件 |
| --- | --- |
| 绝大多数 Windows 电脑 | `MaaShiningNikki-win-x86_64-vXXX.zip` |

解压后运行 `MaaShiningNikki.exe`，选择安卓端控制器并连接模拟器或设备即可。首次启动会根据 `config/pip_config.json` 检查并安装 Agent 依赖。

如果无法启动，请先安装 [`Visual C++ 可再发行程序包`](https://aka.ms/vs/17/release/vc_redist.x64.exe) 和对应 `.NET` 桌面运行时，然后重启电脑。

Windows 10 或 11 用户也可以使用 `winget` 安装常见运行库：

```bash
winget install Microsoft.VCRedist.2015+.x64
```

### macOS / Linux

项目工作流会构建 macOS 与 Linux 包体，但当前主要测试集中在 Windows。若在 macOS 或 Linux 上使用，请参考包内 MFAAvalonia 启动方式，并在反馈问题时附带系统版本、运行日志和截图。

## 开发相关

MaaShiningNikki 基于 [MaaFramework](https://github.com/MaaXYZ/MaaFramework) 开发，可参考 [MaaFramework 官方文档](https://maa.plus/docs/zh-cn/)。

## 免责声明

本软件开源、免费，仅供学习交流使用。若您遇到商家使用本软件收费，产生的费用、问题及后果与本软件无关。

**在使用过程中，MaaShiningNikki 可能存在任何意想不到的问题。因软件漏洞、文本理解歧义、识别错误、异常操作等导致的账号问题或资源损失，开发者不承担任何责任。请在阅读说明并自行确认运行效果后谨慎使用。**

## Star History

<a href="https://www.star-history.com/?repos=7iko-07%2FMaaShiningNikki&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=7iko-07/MaaShiningNikki&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=7iko-07/MaaShiningNikki&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=7iko-07/MaaShiningNikki&type=date&legend=top-left" />
 </picture>
</a>

## 鸣谢

本项目由 **[MaaFramework](https://github.com/MaaXYZ/MaaFramework)** 强力驱动！  
UI 由 [MFAAvalonia](https://github.com/SweetSmellFox/MFAAvalonia) 大力支持！

感谢[MaaGakumasu](https://github.com/SuperWaterGod/MaaGakumasu)提供的Agent实现思路！

感谢[MaaLYSK](https://github.com/Witty36/MaaLYSK)的pipeline架构思路



感谢以下开发者对本项目作出的贡献：

[![Contributors](https://contrib.rocks/image?repo=7iko-07/MaaShiningNikki&max=1000)](https://github.com/7iko-07/MaaShiningNikki/graphs/contributors)
