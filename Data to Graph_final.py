import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from scipy.stats import mannwhitneyu, spearmanr
import pingouin as pg

# 1. 설정: 창 띄우지 않기 및 스타일 정의
matplotlib.use('Agg')
plt.rcParams['font.family'] = 'Pretendard'
plt.rcParams['svg.fonttype'] = 'none'  # 폰트를 깨지 않고 텍스트로 유지 (수정 가능)
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('dark_background')

BG_COLOR = '#1e1f24'
POINT_COLOR = '#FFE000'
GRID_LINE_COLOR = '#ffffff' # 점선용 흰색
AXIS_LINE_COLOR = '#888888' # 축 테두리용 짙은 회색

def apply_custom_style(ax, title, xlabel, ylabel):
    """파란색을 완벽히 제거한 상윤님 포폴 맞춤형 스타일"""
    ax.set_facecolor(BG_COLOR)
    
    # 타이틀 및 라벨 설정
    ax.set_title(title, color='white', pad=25, fontsize=17, fontweight='bold')
    ax.set_xlabel(xlabel, color='#999999', fontsize=8, fontweight='light', labelpad=12)
    ax.set_ylabel(ylabel, color='#999999', fontsize=8, fontweight='light', labelpad=12)
    
    # 위쪽, 오른쪽 테두리 제거
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 왼쪽, 아래쪽 축 선 (무채색 회색으로 설정)
    ax.spines['left'].set_color(AXIS_LINE_COLOR)
    ax.spines['bottom'].set_color(AXIS_LINE_COLOR)
    
    # 축의 눈금(Ticks) 색상도 무채색으로 변경
    ax.tick_params(axis='both', colors='#888888', labelsize=10)
    
    # 얇은 흰색 점선 그리드 (파란기 전혀 없는 순수 화이트)
    ax.grid(True, linestyle=':', linewidth=0.5, color=GRID_LINE_COLOR, alpha=0.15)

# 2. 데이터 로드 및 전처리
df = pd.read_csv('SNS_데이터_분류완료_수정됨(중요).csv')
df['그립톡 유무'] = df['그립톡 갯수'].apply(lambda x: '있음' if x > 0 else '없음')
event_df = df[df['데이터 종류'] == '참여형 이벤트 (댓글 이벤트)'].copy()

# 3. 분석 3, 4, 5, 7: 일반 상관관계 그래프 생성 및 저장
corr_pairs = [
    ('상품 수', '댓글 수'), ('상품 수', '좋아요 수'),
    ('경품 총 금액', '댓글 수'), ('경품 총 금액', '좋아요 수'),
    ('최대 경품 금액', '댓글 수'), ('최대 경품 금액', '좋아요 수'),
    ('그립톡 갯수', '댓글 수'), ('그립톡 갯수', '좋아요 수')
]

for idx, (iv, dv) in enumerate(corr_pairs, 1):
    rho, p = spearmanr(event_df[iv], event_df[dv])
    
    fig, ax = plt.subplots(figsize=(8, 6), facecolor=BG_COLOR)

    # scatter_kws와 line_kws에서 색상을 POINT_COLOR(#FFE000)로 강제 지정하여 파란색 발생 차단
    sns.regplot(x=iv, y=dv, data=event_df, 
                scatter_kws={'s':100, 'color': POINT_COLOR, 'alpha': 0.3}, 
                line_kws={'color': POINT_COLOR, 'linewidth': 2.5}, ax=ax)
    
    # 상단 기술 정보 (Method & p-value)
    info_text = f"Method: Spearman Correlation\np-value: {p:.4f}"
    ax.text(0.05, 0.95, info_text, transform=ax.transAxes, color=POINT_COLOR, 
            fontsize=10, fontweight='light', verticalalignment='top')
    
    apply_custom_style(ax, f'{iv} x {dv} Analysis', iv, dv)
    
    plt.tight_layout()
    plt.savefig(f'분석_결과_{idx}_{iv}_{dv}.svg', format='svg', facecolor=BG_COLOR)
    plt.close()

# 4. 고도화 분석: 편상관분석 그래프 생성 및 저장
# 서열 기반 데이터 변환
event_ranked = event_df[['상품 수', '경품 총 금액', '댓글 수']].rank()

# [상품 수의 독립적 힘 증명] : 금액 통제
partial_res = pg.partial_corr(data=event_ranked, x='상품 수', y='댓글 수', covar='경품 총 금액')
r_val = partial_res['r'].values[0]
p_val = partial_res['p-val'].values[0]

fig, ax = plt.subplots(figsize=(8, 6), facecolor=BG_COLOR)
sns.regplot(x=event_ranked['상품 수'], y=event_ranked['댓글 수'], 
            scatter_kws={'s':120, 'color': POINT_COLOR, 'alpha': 0.4}, 
            line_kws={'color': POINT_COLOR, 'linewidth': 2.5}, ax=ax)

info_text = f"Method: Partial Correlation\nControl: Budget\np-value: {p_val:.4f}"
ax.text(0.05, 0.95, info_text, transform=ax.transAxes, color=POINT_COLOR, 
        fontsize=11, fontweight='bold', verticalalignment='top')

apply_custom_style(ax, 'Partial Correlation: Prize Quantity Influence', 'Rank(Quantity)', 'Rank(Comments)')

plt.tight_layout()
plt.savefig('분석_고도화_편상관분석.svg', format='svg', facecolor=BG_COLOR)
plt.close()

print("="*50)
print("🚀 모든 분석 그래프가 svg로 저장되었습니다. (Pretendard 적용)")
print("="*50)