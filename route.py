# coding=utf-8
import numpy as np
import pandas as pd
import pulp
from pulp import *
from time import time
from tkinter import _flatten
from itertools import product
import re
from pandas import DataFrame


def read_data(file):  # 从数据中提取供给或需求点的坐标
    df = pd.read_csv(file, encoding='utf-8')
    df = df.reset_index(drop=True)
    return df


def deal_od(od_data, data_demand, data_supply):
    num_demand = len(data_demand)
    num_supply = len(data_supply)
    dis = np.array(od_data['distance'])
    dem_index = data_demand.index.tolist()
    sup_index = data_supply.index.tolist()
    dis = dis.reshape(num_demand, num_supply)
    df_od = DataFrame(dis, index=dem_index, columns=sup_index)
    return df_od


def setpath(filepath, H):
    path = os.path.dirname(filepath)
    print(path)
    file_name = os.path.basename(filepath).split('.')[0]
    output_name_assign = file_name + "_" + str(H) + "_assign1351.csv"
    output_name_sites = file_name + "_" + str(H) + "_select1351.csv"
    outputpath_assign = os.path.join(path, output_name_assign)
    outputpath_sites = os.path.join(path, output_name_sites)
    print(outputpath_assign)
    print(outputpath_sites)
    return outputpath_assign, outputpath_sites


def GUROBI(gapRel):
    pass


def P_medine(demands_file, supply_file, od, P, assignresult_path, df_sites_slect_path):
    #       demands_file:需求点数据
    #       supply_file:供给点数据
    #      I：需求点列表
    #      J: 供给店列表
    #     demans:需求列表
    #     facilityCapacity：供给列表
    #     nodedij:OD矩阵
    #     P:需要选择的设施点数量
    demands = demands_file['Num']
    facilityCapacity = supply_file['Num']
    I = demands_file.index.tolist()
    J = supply_file.index.tolist()
    print(J)
    # 建立OD矩阵
    distance = deal_od(od, demands_file, supply_file)
    variables = {}
    # 设置目标函数
    prob = LpProblem("P_medine", LpMinimize)
    # 设置决定变量y_j：0,1变量
    y = LpVariable.dicts("y", J, lowBound=0, upBound=1, cat="Integer")
    #  #设置变量x_ij：整型
    costs = {}
    for i in I:
        for j in J:
            # 各小区被服务的人口
            variables["x_" + str(i) + "_" + str(j)] = pulp.LpVariable("x_" + str(i) + "_" + str(j), 0, None, LpInteger)
            cost = distance[j][i]  # 需求点到供给点的距离（需求点的人口）
            costs["x_" + str(i) + "_" + str(j)] = cost
    # 计算目标值
    obj = 0
    # prob += lpSum([y[j] for j in range(194, 213, 1)]) == 19#保留推荐
    # prob += lpSum([y[j] for j in range(138, 213, 1)]) == 75#四环保留全部75
    # prob += lpSum([y[j] for j in range(208, 228, 1)]) == 20#保留推荐
    # prob += lpSum([y[j] for j in range(149, 228, 1)]) == 79#保留全部
    prob += lpSum([y[j] for j in range(318, 393, 1)]) == 75
    prob += lpSum([costs[x] * variables[x] for x in variables])
    # 设置约束条件：使需求点需求量被一个或多个设施服务
    for i in I:
        s = 0
        for j in J:
            s += variables["x_" + str(i) + "_" + str(j)]
        prob += s == demands[i]
        # print(s)
        # print(demands[i])
    # 设置约束条件：设施服务的需求不超过其供给能力
    for j in J:
        ss = 0
        for i in I:
            ss += variables["x_" + str(i) + "_" + str(j)]
        prob += ss <= facilityCapacity[j] * y[j]
    # 设施约束条件被选择设施总数=P
    prob += lpSum([y[j] for j in J]) == P
    # 输出lp文件
    prob.writeLP("_tp.lp")
    # 设置求解器
    # solver=GUROBI()
    solver = GUROBI(gapRel=0.05)
    # 模型求解
    # 初始化求解时间
    time_start = time()
    prob.solve(solver)
    time_end = time()
    tot_cost = str(time_end - time_start)
    result_alert = "*求解状态:" + str(LpStatus[prob.status]) + "\n" + "*Total Cost of The Model =" + str(
        value(prob.objective)) + "\n*模型求解时间为：" + tot_cost
    # print(prob.variables())
    if prob.status < 0:
        print("模型求解失败")
    else:
        sites_slect = []
        assign_lists = []
        for v in prob.variables():
            if v.varValue >= 1:
                if re.match('y_', v.name) is not None:
                    items = v.name.split('_')
                    j = int(items[1])
                    attribute_j = supply_file.iloc[j].tolist()
                    attribute_j.insert(0, j)
                    sites_slect.append(attribute_j)
                else:
                    assign_list = []
                    items = v.name.split('_')
                    i = int(items[1])
                    k = int(items[2])
                    assign = v.varValue
                    assign_list.append(i)
                    attribute_i = demands_file.loc[i, ['Num', 'Name', 'Latitude', 'Longitude']].to_list()
                    assign_list.append(attribute_i)
                    assign_list.append(k)
                    attribute_k = supply_file.loc[k, ['Num', 'Latitude', 'Longitude']].to_list()
                    assign_list.append(attribute_k)
                    assign_list.append(assign)
                    assign_list.append(distance[k][i])
                    assign_lists.append(list(_flatten(assign_list)))
        try:
            assignresult = DataFrame(assign_lists,
                                     columns=["i", 'Num', 'Name', 'Latitude_d', 'Longitude_d', "k", 'Num', 'Latitude_s',
                                              'Longitude_s', "assign", "distance"])
            df_sites_slect = DataFrame(sites_slect, columns=['index_selected', 'Num', 'Longitude', 'Latitude'])
            # assignresult_path = "E:\\AND NEW\\data\\Arcgis\\候选点的选取\\p_median\\gap1\\assignresult_g1_a35test.csv"
            # df_sites_slect_path = "E:\\AND NEW\\data\\Arcgis\\候选点的选取\\p_median\\gap1\\df_sites_slect_g1_a35test.csv"
            # assignresult_path =r"G:data\assignresult.csv"
            # df_sites_slect_path =r'G:\data\slect_135.csv'
            assignresult.to_csv(assignresult_path, index=False, encoding='utf-8')
            print("assignresult:" + assignresult_path)
            df_sites_slect.to_csv(df_sites_slect_path, index=False, encoding='utf-8')
            print("df_sites_slect:" + df_sites_slect_path)
        except:
            print("文件下写入失败")
    print(df_sites_slect)
    print(assignresult)
    print(result_alert)
    # return assignresult


if __name__ == '__main__':
    # demands_file = r'G:\data\实地调研\数据整理结果\处理结果\模型数据\zz市小区_分配人口.csv'
    # sites_file = r"G:\data\实地调研\数据整理结果\处理结果\模型数据\候选点\全部候点\保留推荐标准+新增\保留推荐标准_新增_国家标准_100000.csv"
    # od_file = r"G:\data\实地调研\数据整理结果\处理结果\模型数据\
    # OD计算\候选点OD计算\zz市保留推荐+新增候选点_OD.csv"
    demands_file = r"G:\data\实地调研\数据整理结果\四环\模型数据\四环小区.csv"
    sites_file = r"G:\data\实地调研\数据整理结果\四环\模型数据\四环候选点318.csv"
    od_file = r"G:\data\实地调研\数据整理结果\四环\模型数据\四环候选点OD318sort.csv"
    demands = read_data(demands_file)  # demand nodes
    sites = read_data(sites_file)  # facility sites
    od = read_data(od_file)  # od 矩阵
    for H in range(135, 145, 10):
        assign_path, slect_path = setpath(sites_file, H)
        P_medine(demands, sites, od, H, assign_path, slect_path)
