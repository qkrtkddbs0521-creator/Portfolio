import pandas as pd
import re
import random
import matplotlib.pyplot as plt
from collections import defaultdict
from difflib import SequenceMatcher
import numpy as np

def remove_emoji(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F700-\U0001F77F"  # alchemical symbols
                               u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                               u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                               u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                               u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                               u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                               u"\U00002702-\U000027B0"  # Dingbats
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    cleaned_text = emoji_pattern.sub(r'', str(text))
    cleaned_text = cleaned_text.strip()
    return cleaned_text

def calculate_score(row, length_weight=1.0, bonus_weight=1.0, base_score=0.0):
    text_to_check = ''.join(str(val) for val in row)
    cleaned_text = remove_emoji(text_to_check)
    
    if cleaned_text.strip() == '':
        return 0

    matches = re.findall(r'@(\w{3,})', cleaned_text)
    bonus_score = 2 * len(set(matches)) * bonus_weight
    length_score = (len(cleaned_text) / 30 * length_weight)
    total_score = length_score + bonus_score + base_score
    return total_score


def get_exclude_users(exclude_users_file):
    if exclude_users_file is None:
        return []

    try:
        if exclude_users_file.endswith('.csv'):
            exclude_users_df = pd.read_csv(exclude_users_file, encoding='utf-8')
        elif exclude_users_file.endswith('.xlsx'):
            exclude_users_df = pd.read_excel(exclude_users_file, engine='openpyxl')
        else:
            raise ValueError("지원하지 않는 파일 형식입니다.")

        if '이름' in exclude_users_df.columns:
            exclude_users_list = exclude_users_df['이름'].astype(str).str.strip().tolist()
            return exclude_users_list
        else:
            return []

    except FileNotFoundError as e:
        print(f"파일을 찾을 수 없습니다: {e}")
        return []
    except pd.errors.EmptyDataError as e:
        print(f"데이터가 비어있습니다: {e}")
        return []
    except Exception as e:
        print(f"파일을 읽는 중 오류가 발생했습니다: {e}")
        return []

def find_suspicious_accounts(df, name_column):
    suspicious_accounts = defaultdict(list)
    name_to_accounts = defaultdict(set)

    combined_text = df.apply(lambda row: ' '.join(map(str, row)), axis=1)
    cleaned_text = combined_text.apply(remove_emoji).astype(str).str.strip()

    for index, text in cleaned_text.items():
        matches = re.findall(r'@(\w{3,})', text)
        if matches:
            unique_matches = set(matches)
            for account in unique_matches:
                if pd.notnull(df.loc[index, name_column]):
                    suspicious_accounts[account].append(df.loc[index, name_column])
                    name_to_accounts[df.loc[index, name_column]].add(account)

    output_file = 'suspicious_accounts.xlsx'
    grouped_data = []
    suspicious_account_set = set()

    for account, names in suspicious_accounts.items():
        if len(names) >= 3:
            print(f"계정 '{account}'이(가) 의심 계정으로 분류되었습니다.")
            print("관련된 이름:", ', '.join(map(str, set(names))))
            suspicious_account_set.add(account)

    for name, associated_accounts in name_to_accounts.items():
        associated_accounts_str = ', '.join(sorted(associated_accounts))
        grouped_data.append({'계정': name, '언급된 이름': associated_accounts_str})

    grouped_df = pd.DataFrame(grouped_data)
    grouped_df.to_excel(output_file, index=False)
    print("그룹화된 계정 정보를 'suspicious_accounts.xlsx' 파일로 저장했습니다.")

    return suspicious_account_set

    def detect_cycles(account_graph):
        visited = set()
        stack = set()

        def visit(account):
            if account in visited:
                return False
            visited.add(account)
            stack.add(account)
            for neighbor in account_graph[account]:
                if neighbor in stack or visit(neighbor):
                    return True
            stack.remove(account)
            return False

        return any(visit(account) for account in account_graph)

    account_graph = defaultdict(set)
    for name, mentions in account_mentions.items():
        for mention in mentions:
            account_graph[name].add(mention)

    if detect_cycles(account_graph):
        print("언급 순환 패턴이 감지되었습니다.")
        for account in account_graph:
            suspicious_account_set.add(account)

    return suspicious_account_set

def select_winners(file_path, num_prizes, name_column, exclude_users_file=None, length_weight=1.0, bonus_weight=1.0, log_base=10):
    exclude_users = get_exclude_users(exclude_users_file)
    print("=== 현재 exclude_users ===")
    print(exclude_users)
    winners = {}

    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path, encoding='utf-8')
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path, engine='openpyxl')
        else:
            raise ValueError("지원하지 않는 파일 형식입니다.")

        df = df.dropna(subset=[name_column])
        df = df[df[name_column].astype(str).str.strip() != '']

        suspicious_accounts = find_suspicious_accounts(df, name_column)

        df['스코어'] = df.apply(lambda row: calculate_score(row, length_weight, bonus_weight), axis=1)
        eligible_rows = df[df['스코어'] >= 0]

        eligible_rows = eligible_rows[~eligible_rows[name_column].isin(suspicious_accounts)]
        eligible_rows = eligible_rows[~eligible_rows[name_column].isin(exclude_users)]

        eligible_rows['로그_스코어'] = np.log(eligible_rows['스코어'] + 1) / np.log(log_base)
        total_log_score = eligible_rows['로그_스코어'].sum()

        selected_indices = set()
        while len(selected_indices) < num_prizes and not eligible_rows.empty:
            selected_index = random.choices(
                eligible_rows.index,
                weights=eligible_rows['로그_스코어'],
                k=1
            )[0]
            selected_indices.add(selected_index)
            eligible_rows = eligible_rows.drop(selected_index)

        for index in selected_indices:
            row = df.loc[index]
            winner_name = row[name_column]
            score = row['스코어']
            winners[winner_name] = {
                '이름': winner_name,
                '등수': len(winners) + 1,
                '스코어': score,
                '로그_스코어': np.log(score + 1) / np.log(log_base),
                '당첨확률': (np.log(score + 1) / np.log(log_base)) / total_log_score
            }

    except Exception as e:
        print(f"파일을 읽는 중 오류가 발생했습니다: {e}")
        return None

    for winner_name in winners:
        winners[winner_name]['이름'] = f"@{winners[winner_name]['이름']}"

    return winners

def plot_quartiles_and_save(df, winners, name_column, output_file='winners_quartiles.jpg'):
    try:
        df['사분위'] = pd.qcut(df['스코어'], 4, labels=False, duplicates='drop')
    except ValueError:
        print("Data variation too low to form quartiles. Plotting all data in a single box.")
        df['사분위'] = 0  # Assign all data to one bin

    winner_scores = [winners[winner]['스코어'] for winner in winners]

    plt.figure(figsize=(10, 6))
    plt.boxplot(df['스코어'], vert=False, patch_artist=True)
    plt.scatter(winner_scores, [1] * len(winner_scores), color='red', zorder=2)

    plt.title('Winner Scores on Score Distribution')
    plt.xlabel('Score')
    plt.yticks([])
    plt.savefig(output_file)
    plt.close()
    print(f"사분위수 시각화를 {output_file}로 저장했습니다.")
def plot_log_score_vs_rank(winners, output_file='log_score_vs_rank.jpg'):
    try:
        winner_names = list(winners.keys())
        ranks = [winners[name]['등수'] for name in winner_names]
        log_scores = [winners[name]['로그_스코어'] for name in winner_names]

        plt.figure(figsize=(10, 6))
        plt.scatter(log_scores, ranks, color='blue', zorder=2)
        plt.title('Rank vs Log Score')
        plt.xlabel('Log Score')
        plt.ylabel('Rank')
        plt.gca().invert_yaxis()  # Ensure that rank 1 is at the top
        plt.grid(True)
        plt.savefig(output_file)
        plt.close()
        print(f"로그 스코어와 등수에 따른 산점도를 {output_file}로 저장했습니다.")
    except Exception as e:
        print(f"산점도를 그리는 중 오류가 발생했습니다: {e}")

def main():
    file_path = input("이벤트 참가자 정보가 있는 파일의 경로를 입력하세요: ")
    exclude_users_file = input("제외할 참가자 목록이 있는 엑셀 파일의 경로를 입력하세요: ")
    num_prizes = int(input("이벤트 상품의 등수를 입력하세요: "))
    name_column = input("참가자의 이름이 들어있는 열의 이름을 입력하세요: ")
    length_weight = float(input("텍스트 길이에 대한 가중치를 입력하세요 (기본값은 1.0): ") or 1.0)
    bonus_weight = float(input("보너스 점수에 대한 가중치를 입력하세요 (기본값은 1.0): ") or 1.0)
    log_base = float(input("로그 함수의 밑 값을 입력하세요 (기본값은 10): ") or 10)

    winners = select_winners(file_path, num_prizes, name_column, exclude_users_file=exclude_users_file, length_weight=length_weight, bonus_weight=bonus_weight, log_base=log_base)

    if winners:
        result_df = pd.DataFrame(list(winners.items()), columns=['이름', '정보'])
        result_df[['이름', '등수', '스코어', '로그_스코어', '당첨확률']] = result_df['정보'].apply(pd.Series)
        result_df.drop(['정보'], axis=1, inplace=True)
        result_df.sort_values(by=['등수'], inplace=True)

        result_df.to_excel('winners_result.xlsx', index=False)
        print("결과를 'winners_result.xlsx' 파일로 저장했습니다.")

        # Ensure to pass the correct DataFrame with '스코어' to the plot function
        plot_quartiles_and_save(result_df, winners, name_column)
        
        # Plot log score vs rank scatter plot
        plot_log_score_vs_rank(winners)

if __name__ == "__main__":
    main()