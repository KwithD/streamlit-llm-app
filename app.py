from dotenv import load_dotenv
load_dotenv(".env", override=True)  # ローカルでは .env から OPENAI_API_KEY を読む

import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# ---------------------------
# 使い方テキスト（概要・操作方法）
# ---------------------------
st.set_page_config(page_title="LangChain × Streamlit LLM App")
st.title("Streamlit課題アプリ（観光業界）")

st.markdown(
    """
**概要**  
入力テキストを LangChain 経由で LLM に渡し、結果を画面に表示します。  
ラジオボタンで「専門家の振る舞い（システムメッセージ）」を選べます。

**操作方法**  
1. 専門家の種類を選ぶ（航空／鉄道／ホテル／自動車）  
2. テキストを入力  
3. 「送信」を押す → 回答が表示
"""
)

# ---------------------------
# 専門家の役割（A/B→業界4種に変更）
# ---------------------------
EXPERT_ROLES = {
    "航空業界の専門家": (
        "あなたは航空業界の専門家です。需要予測（季節性・路線特性）、収益管理（運賃クラス・座席販売最適化）、"
        "運航計画（機材繰り・クルースケジュール）、旅客体験（NPS/CS）、アライアンス/コードシェア、空港発着枠、"
        "付帯収益（手荷物・座席指定・ラウンジ）を踏まえ、ユーザーの課題に対し、"
        "1) 背景・論点, 2) 実務的な施策（3案以内）, 3) KPI/次アクション を日本語で簡潔に提案してください。"
    ),
    "鉄道業界の専門家": (
        "あなたは鉄道業界の専門家です。需要の時間帯・線区別特性、運賃体系、列車運行（ダイヤ/折返し/運用）、"
        "保守計画（車両/設備）、安全・遅延要因、駅商業/広告/IC連携、沿線開発・不動産収益を踏まえて、"
        "ユーザーの課題に対し、1) 背景・制約, 2) 実行可能な施策（3案以内）, 3) KPI/ステークホルダー整理 を提示してください。"
    ),
    "ホテル業界の専門家": (
        "あなたはホテル業界の専門家です。ADR/RevPAR/稼働率、シーズナリティとイベント需要、OTA/直販チャネル、"
        "客室タイプ構成、清掃・人員シフト、F&B/宴会/付帯収益、レビュー・NPS改善を考慮し、"
        "ユーザーの課題に対し、1) 需要/供給の見立て, 2) 料金/在庫/販路最適化（3案以内）, 3) 運営KPI/次アクション を出してください。"
    ),
    "自動車業界の専門家": (
        "あなたは自動車業界の専門家です。製品企画（EV/HEV/ICE/商用）、サプライチェーン（調達/在庫）、"
        "販売チャネル（ディーラー/オンライン）、アフターサービス、コネクテッド/ソフトウェア、"
        "規制・安全基準・補助金、市場別戦略を踏まえ、ユーザーの課題に対して、"
        "1) 前提と仮説, 2) 実行施策（3案以内）, 3) 成功指標/リスクと緩和策 を日本語で簡潔に提案してください。"
    ),
}

# ---------------------------
# LLM呼び出し用 関数（要件：引数=入力テキスト/選択値、戻り値=回答）
# ---------------------------
def ask_llm(user_text: str, expert_key: str) -> str:
    """選択された専門家プロンプトで LLM 応答を返す"""
    system_msg = EXPERT_ROLES.get(expert_key, list(EXPERT_ROLES.values())[0])

    # LangChain: Prompt → LLM → Parser のシンプルなチェーン
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "{system_msg}"),
            ("user", "{user_text}"),
        ]
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",  # 例：軽量で高速。必要に応じて変更可
        temperature=0.2,
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"system_msg": system_msg, "user_text": user_text})

# ---------------------------
# UI：専門家選択 + 入力フォーム
# ---------------------------
expert_choice = st.radio("専門家の種類を選択", list(EXPERT_ROLES.keys()))
user_input = st.text_area("入力テキスト", value="繁忙期の需要に合わせて価格と在庫を最適化したい。初期ステップを教えて。")

if st.button("送信"):
    if not user_input.strip():
        st.warning("入力テキストを入れてください。")
    else:
        with st.spinner("LLMに問い合わせ中…"):
            try:
                answer = ask_llm(user_input, expert_choice)
                st.subheader("回答")
                st.write(answer)
            except Exception as e:
                st.error(f"エラー: {e}")



