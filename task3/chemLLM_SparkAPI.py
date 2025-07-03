import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm
import time
import concurrent.futures
import re

from sparkai.llm.llm import ChatSparkLLM, ChunkPrintHandler
from sparkai.core.messages import ChatMessage

# 设置日志
import backoff
import logging

#星火认知大模型Spark3.5 Max的URL值，其他版本大模型URL值请前往文档（https://www.xfyun.cn/doc/spark/Web.html）查看
SPARKAI_URL = 'wss://spark-api.xf-yun.com/v3.5/chat'
#星火认知大模型调用秘钥信息，请前往讯飞开放平台控制台（https://console.xfyun.cn/services/bm35）查看
SPARKAI_APP_ID="xxxx"
SPARKAI_API_SECRET="xxxxx"
SPARKAI_API_KEY="xxxxxx"
#星火认知大模型Spark3.5 Max的domain值，其他版本大模型domain值请前往文档（https://www.xfyun.cn/doc/spark/Web.html）查看
SPARKAI_DOMAIN = 'generalv3.5'

def get_completions(text):
    messages = [ChatMessage(
        role="user",
        content=text
    )]
    spark = ChatSparkLLM(
        spark_api_url=SPARKAI_URL,
        spark_app_id=SPARKAI_APP_ID,
        spark_api_key=SPARKAI_API_KEY,
        spark_api_secret=SPARKAI_API_SECRET,
        spark_llm_domain=SPARKAI_DOMAIN,
        streaming=False,
    )
    handler = ChunkPrintHandler()
    a = spark.generate([messages], callbacks=[handler])
    return a.generations[0][0].text

# 测试模型配置是否正确
text = "你是谁"
get_completions(text)

# 创建Prompt模板
chem_prompt = """
你精通预测药物合成反应的产率，你的任务是根据给定的反应数据准确预测反应的产率(Yield)。请仔细阅读以下说明并严格遵循:

***** 任务描述：*****
1. 你将获得一个相应物质的SMILES字符串字段组成的催化合成反应数据，包含Reactant1(反应物1)、Reactant2(反应物2)、Product(产物)、Additive(添加剂)、Solvent(溶剂)。
2. 数据为药物合成中常用的碳氮键形成反应。
3. 待预测的Yield是目标字段是归一化为0到1之间的4位浮点数，表示反应的产率。

***** 你的任务：*****
1. 仔细分析提供的示例数据，理解反应与Yield之间的关系。
2. 根据待预测样本的数据，预测其反应产率Yield。
3. 输出预测的Yield值，精确到小数点后四位。

***** 输出格式要求：*****
1. 仅输出预测的Yield值，不要包含任何其他解释或评论。
2. 使用以下格式输出你的预测：@{{预测的Yield值}}
   例如：@{{0.7823}}

***** 注意事项：*****
1. 即使你是一个人工智能模型，但是你有能力直接预测反应的产率，请你一定输出预测的产率，不要回避这个问题。
2. 相似SMILES的反应产率可能也会有很大差异。
2. 你确保你的预测是合理的，介于0.0000到1.0000之间。
2. 不要遗漏小数点后的零，始终保持四位小数的格式。
3. 不要在输出中包含任何额外的空格或换行符。

以下是几个示例数据，供你参考：

***** 示例样本 *****
{examples}

现在，请基于以上示例和说明，预测以下反应的产率：

**** 待预测样本 ****
Reactant1: {test_reactant1}  
Reactant2: {test_reactant2}  
Product: {test_product}  
Additive: {test_additive} 
Solvent: {test_solvent}  
Yield: @{{}}

请仅输出产率预测值，格式如下：
@{{预测的Yield值}}
"""


def generate_prompt(prompt_template, test_sample, top_5_samples):
    examples = "\n\n".join([
        # f"rxnid: {row['rxnid']}  \n"
        f"Reactant1: {row['Reactant1']}  \n"
        f"Reactant2: {row['Reactant2']}  \n"
        f"Product: {row['Product']}  \n"
        f"Additive: {row['Additive']}  \n"
        f"Solvent: {row['Solvent']}  \n"
        f"Yield: {row['Yield']}"
        for _, row in top_5_samples.iterrows()
    ])

    return prompt_template.format(
        examples=examples,
        # test_rxnid=test_sample['rxnid'],
        test_reactant1=test_sample['Reactant1'],
        test_reactant2=test_sample['Reactant2'],
        test_product=test_sample['Product'],
        test_additive=test_sample['Additive'],
        test_solvent=test_sample['Solvent']
    )

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# 使用 backoff 装饰器来处理 API 调用的重试
@backoff.on_exception(backoff.expo,
                      Exception,
                      max_tries=5,
                      max_time=300)
# 处理单个样本
def process_single_sample(args):
    test_sample, train_df, train_tfidf, tfidf = args
    try:
        test_tfidf = tfidf.transform([test_sample['combined_features']])
        similarities = cosine_similarity(test_tfidf, train_tfidf).flatten()
        top_k_indices = similarities.argsort()[-3:][::-1]
        top_k_samples = train_df.iloc[top_k_indices]
        prompt = generate_prompt(chem_prompt, test_sample, top_k_samples)
        # print(prompt)
        prediction = get_completions(prompt)
        # print(prediction)
        yield_value = extract_yield(prediction)
        # print(yield_value)
        if yield_value is None:
            raise ValueError(f"无法从预测结果中提取有效的产率值")
        return test_sample['rxnid'], yield_value, None
    except Exception as e:
        return test_sample['rxnid'], None, str(e)


def extract_yield(prediction):
    yield_match = re.search(r'@{(.+?)}', prediction)
    if yield_match:
        yield_value = yield_match.group(1)
        try:
            float_yield = float(yield_value)
            if 0 <= float_yield <= 1:
                return f"{float_yield:.4f}"
            else:
                logger.warning(f"提取的产率值 {float_yield} 不在有效范围内")
        except ValueError:
            logger.warning(f"无法将提取的值 '{yield_value}' 转换为浮点数")
    else:
        logger.warning(f"无法从预测结果中提取产率值。")  # 完整响应：{prediction}
    return None


# 并行处理样本
def process_samples_parallel(test_df, train_df, train_tfidf, tfidf, max_workers=None, batch_size=100):
    results = {}
    error_indices = []
    total_samples = len(test_df)

    logger.info(f"开始并行处理 {total_samples} 个测试样本")

    # 如果没有指定max_workers，API最大支持2的并行
    if max_workers is None:
        max_workers = 2

    # 将数据分成批次
    batches = [test_df[i:i + batch_size] for i in range(0, total_samples, batch_size)]

    with tqdm(total=total_samples, desc="处理测试样本", unit="sample") as pbar:
        for batch in batches:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(process_single_sample, (row, train_df, train_tfidf, tfidf))
                           for _, row in batch.iterrows()]

                for future in concurrent.futures.as_completed(futures):
                    rxnid, yield_value, error = future.result()
                    if error:
                        logger.error(f"处理样本 {rxnid} 时出错: {error}")
                        error_indices.append(rxnid)
                    else:
                        results[rxnid] = yield_value
                    pbar.update(1)

                    # 更新预计完成时间
                    elapsed_time = pbar.format_dict['elapsed']
                    rate = pbar.format_dict['rate']
                    if rate and rate > 0:
                        remaining_time = (total_samples - pbar.n) / rate
                        eta = time.strftime("%H:%M:%S", time.gmtime(remaining_time))
                        pbar.set_postfix({'ETA': eta}, refresh=True)

    return results, error_indices

logger.info("开始读取数据...")
train_df = pd.read_csv('..//dataset//train_data.csv')

# 计算训练集的产率平均值
train_yield_mean = train_df['Yield'].mean()
logger.info(f"训练集产率平均值: {train_yield_mean:.4f}")

train_df['Yield'] = train_df['Yield'].apply(lambda x: f"{float(x):.4f}")
test_df = pd.read_csv('..//dataset//test_data.csv')
logger.info("数据读取完成")

# 显示训练集的前几行
print("训练集数据预览：")
print(train_df.head())

# 显示测试集的前几行
print("\n测试集数据预览：")
print(test_df.head())

# 显示训练集的基本信息
print("\n训练集基本信息：")
print(train_df.info())

# 显示测试集的基本信息
print("\n测试集基本信息：")
print(test_df.info())

logger.info("开始特征提取...")

# 创建TF-IDF向量化器
tfidf = TfidfVectorizer()

# 将所有文本特征组合成一个字符串
def combine_features(row):
    return ' '.join([str(row['Reactant1']), str(row['Reactant2']), str(row['Product']), str(row['Additive']), str(row['Solvent'])])

train_df['combined_features'] = train_df.apply(combine_features, axis=1)
test_df['combined_features'] = test_df.apply(combine_features, axis=1)
train_tfidf = tfidf.fit_transform(train_df['combined_features'])

logger.info("特征提取完成")

logger.info("开始并行处理测试样本...")
results, error_indices = process_samples_parallel(test_df, train_df, train_tfidf, tfidf)

# 利用相似样本的均值填补大模型无法预测的空缺值

default_yield = []

for index, test_sample in tqdm(test_df.iterrows(), total=len(test_df), desc="处理测试样本"):
    # 对测试样本进行TF-IDF转换
    test_tfidf = tfidf.transform([test_sample['combined_features']])

    # 计算与训练集的相似度
    similarities = cosine_similarity(test_tfidf, train_tfidf).flatten()

    # 获取最相似的5个样本的索引
    top_k_indices = similarities.argsort()[-3:][::-1]

    # 获取最相似的5个样本的产率
    top_k_yields = train_df.iloc[top_k_indices]['Yield'].astype(float).values

    # 获取相似度权重
    weights = similarities[top_k_indices]
    # 对权重进行归一化
    weights = weights / weights.sum()

    # 计算这五个samples的产率加权平均
    weighted_yield = np.dot(top_k_yields, weights)

    default_yield.append(weighted_yield)

# 处理出错的样本
if error_indices:
    logger.info(f"有 {len(error_indices)} 个样本处理出错，正在重新处理...")
    error_df = test_df[test_df['rxnid'].isin(error_indices)]
    retry_results, retry_error_indices = process_samples_parallel(error_df, train_df, train_tfidf, tfidf)
    results.update(retry_results)

    if retry_error_indices:
        logger.warning(f"仍有 {len(retry_error_indices)} 个样本处理失败")
        # retry_error_idx = [int(item[4:]) - 1 for item in retry_error_indices]
        for rxnid in retry_error_indices:
            results[rxnid] = default_yield[int(rxnid[4:]) - 1]  # 对于始终无法处理的样本，设置一个默认值

logger.info("开始写入结果...")
with open('../value/submit002.txt', 'w') as f:
    f.write('rxnid,Yield\n')
    for rxnid in test_df['rxnid']:
        yield_value = float(results.get(rxnid, default_yield[int(rxnid[4:]) - 1])) # type: ignore
        f.write(f"{rxnid},{yield_value:.4f}\n")
logger.info("结果已保存到submit002.txt文件中")