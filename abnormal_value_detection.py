from typing import List, Dict
from datetime import datetime,timedelta
import json
import ast
import sys
import numpy as np
class abnormal_detection:
    #每天体重相差合理范围,公斤
    delt_y = 1.5
    #按x轴分类，类之间距离 ，大于2则单独形成一类
    distance_x = 2
    def __init__(self,delt_y:float , distance_x:str )-> None:
        if not  delt_y is None and  delt_y >0 :
            self.delt_y = delt_y
        if not  distance_x is None :
            self.distance_x = distance_x
    #定义点包含的维度，(x,y,isconfidence)  x坐标刻度 值 有是否可信 isconfidence
    class Point():
        def __init__(self,x:str,y:str,isconfidence:bool=False) -> None:
            self.x = x
            self.y = y
            self.isconfidence = ast.literal_eval(isconfidence.capitalize())
        def __str__(self):
            return f" x  value: {self.x} ,y value : {self.y}  , isconfidence value : {self.isconfidence} "
        
    #依据x轴分类，返回所有分类，key 取此分类 x轴最小值
    def class_by_x(self, value:List[Point]) -> dict:
        # 对List 进行排序，按照x轴 由小到大排序
        value.sort(key=lambda e: e.x)
        print(value)
        #前一个元素，赋初始值
        pre = None
        #当前元素赋初始值
        current = None
        #前一个分类的主键
        pre_key = None
        format = "%Y-%m-%d"
        #返回值
        results = {}
        for item in value:
            current = item
            #如果是列表的第一个元素，将前一个元素和当前元素都赋值为 第一个元素值
            if pre is None:
                pre = item
                #设定第一个分类的主键 ，并生成第一个分类
                pre_key=pre.x
                results[pre_key] = []
                results[pre_key].append(pre)
                continue
            #获取当前元素和前面一个元素的x轴值 并转换为日期格式
            current_date = datetime.strptime(current.x, format)
            pre_date = datetime.strptime(pre.x, format)
            #如果当前元素x轴的日期与上个元素日期距离大于阈值 self.distance_x ， 则产生新的分类 ，新分类第一个元素为当前值，主键为第一个元素的日期
            if(current_date >  pre_date +timedelta(days=self.distance_x)):
                results[current.x]=[]
                results[current.x].append(current)
                pre_key = current.x
            # 如果当前元素和上一个元素属于同一个分类，则把其添加上个分类中     
            else:
                results[pre_key].append(current)
            #更新上个元素值， 为下个循环做准备
            pre = item
        return results
    
    def pareJson(self , json_weights :str) -> List[Point]:
        if json_weights is None:
            return None
        data = json.loads(json_weights)
        result = []
        for item in data:
            if(item['date'] is None or item['value'] is None):
                print("数据值为None,跳过处理")
                continue
            p = self.Point(item['date'],item['value'],item['isconfidence'])
            result.append(p)
        return result

    def deal(self, json_weights:str) -> List:
        points = self.pareJson(json_weights)
        class_result = self.class_by_x(points)
        excep = self.detction(class_result)
        print (excep)
        return excep

    # 每个分类里面找到一个可信值，一个分类里面包含2个以上的点才找可信值，少于3个的 ，不补充可信值 ,分类 元素日期间隔不超过 3天 ，一个类里面数值连续日期的值
    def addConfidence(self,class_result:dict[str,List[Point]]) ->None:
        for key ,value in class_result.items():
            if(len(value) < 3):
                continue
            #如果此类里面数据包含大于等于3个数据点，取出所有数值
            weight =[]
            #是否存在可信值
            isconfidence=False
            for p in value:
                if(p.isconfidence):
                    isconfidence =p.isconfidence
                    weight.clear()
                    break
                weight.append(p.y)
            #如果不存在信赖值 ，系统把第二百分位数设为可信数
            if(not isconfidence):
                data = np.array(weight)
                # 计算第25百分位数（第一个四分位数）
                Q1 = np.percentile(data, 25)            
                # 计算中位数（第二个四分位数）
                Q2 = np.percentile(data, 50)           
                # 计算第75百分位数（第三个四分位数）
                Q3 = np.percentile(data, 75)
                #在数据中找到 最接近50分位点的数
                isFound = False
                sortedValue = sorted(value,key=lambda vl: vl.y)
                for v in sortedValue:
                    if( v.y >= Q2 ):
                        for it in value:
                            if (it.y == v.y):
                                it.isconfidence = True
                                isFound =True
                                break
                        break
                if(not isFound):
                    print("注意：没有找到大于等于50分为点的数据，数据长度为：" +len(sortedValue) )
                        
        return None
    # 在一个升序的list 中找到  input 字符串的邻居
    def find_neighbor(self,input:str, sorted_findList:List[str]) -> str:
       find = None
       for v in sorted_findList:
           find = v
           if (input >= v ):
               continue
       return find
           

    #针对已经分类的数值识别异常点
    def detction(self,class_result:dict[str,List[Point]]) -> List[Point]:
        results =[]
        nofFondconf = {}
        sortedfondKeys=[]
        self.addConfidence(class_result)
        for key,value in class_result.items():
            isFindConf = False
            #在分类里面找可信值，如果找到一个，则处理异常值，后面不再继续找
            for index  in range(len(value)):
                if(value[index].isconfidence):
                    isFindConf =True
                    sortedfondKeys.append(key)
                    res = self.findException(index,value)
                    results.extend(res)
                    break
            #如果本分类里面没有可信值，则记录下来
            if(not isFindConf):
                nofFondconf[key] = value
        
        if(len(nofFondconf) >0 and len(sortedfondKeys) >0):
            for key, v in nofFondconf.items():
                # 找到邻居，并取邻居里与其最靠近的一条可信数据，作为参照对象
                neighbor = self.find_neighbor(key,sortedfondKeys)
                for item in class_result[neighbor]:
                    if(item.isconfidence):
                        v.append(item)
                        break
                    else:
                        print("没有找到邻居中，可信的数据作为参照，请检查数据,系统将退出。")
                        sys.exit()
                
                res = self.findException(len(v)-1,v)
                results.extend(res)
        else:
            if(len(nofFondconf) <=0):
                pass
            else:
                #都没有可信值,把分类合并
                totalList = []
                for k ,v in class_result.items():
                    totalList.extend(v)
                weights =[]
                for p in totalList:
                    weights.append(p.y)
                data = np.array(weights)
                # 计算第25百分位数（第一个四分位数）
                Q1 = np.percentile(data, 25)            
                # 计算中位数（第二个四分位数）
                Q2 = np.percentile(data, 50)           
                # 计算第75百分位数（第三个四分位数）
                Q3 = np.percentile(data, 75)

                sortedValue = sorted(totalList,key=lambda vl: vl.y)
                isFound = False
                for i in range(len(sortedValue)):
                    if( sortedValue[i].y >= Q2 ):
                        sortedValue[i].isconfidence = True
                        isFound = True
                        res = self.findException(i,sortedValue)
                        results.extend(res)
                        break
                if(not isFound):
                    print("注意：没有找到大于等于50分为点的数据，数据长度为：" +len(sortedValue) )

        return results
    
        


    def findException(self,index:int , values: List[Point]) -> List[Point]:
        ConfidenceP= values[index]
        format  = "%Y-%m-%d"
        exceptionList=[]
        for i in range(index+1,len(values)):
            p = values[i]
            conf_date = datetime.strptime(ConfidenceP.x, format)
            p_date =  datetime.strptime(p.x, format)
            delt = p_date -conf_date
            predit_max = delt.days * self.delt_y + ConfidenceP.y
            predit_min = ConfidenceP.y - delt.days * self.delt_y
            if( p.y > predit_max or p.y < predit_min):
                if(not p.isconfidence):
                    print("发现异常点：" +p.__str__())
                    exceptionList.append(p)
                else:
                    print(f"按照 每天体重变化{self.delt_y}公斤 ，发现异常值：" +p.__str__() )
                    print("参照对下为：" +ConfidenceP.__str__() + " , 属于异常值，但此异常点可信度为真，因此采纳此值 ")
                    ConfidenceP = p
            else:
                p.isconfidence = True
                ConfidenceP = p

        


        ConfidenceP= values[index]
        for j  in range(index-1,-1,-1):
            p = values[j]
            conf_date = datetime.strptime(ConfidenceP.x, format)
            p_date =  datetime.strptime(p.x, format)
            delt = p_date -conf_date
            predit_min = delt.days * self.delt_y+ ConfidenceP.y
            predit_max = ConfidenceP.y - delt.days * self.delt_y
            if( p.y > predit_max or p.y < predit_min):
                if(not p.isconfidence):
                    print(f"按照 每天体重变化{self.delt_y}公斤 ，发现异常点：" +p.__str__())
                    exceptionList.append(p)
                else:
                    print("按照 每天体重变化{self.delt_y}公斤 ，发现异常值：" +p.__str__() )
                    print("参照对下为：" +ConfidenceP.__str__() + " , 属于异常值，但此异常点可信度为真，因此采纳此值 ")
                    ConfidenceP = p
            else:
                p.isconfidence = True
                ConfidenceP = p

        return exceptionList
           
if __name__ == '__main__' :
    print("This script is running as the main program.")
    detection = abnormal_detection(1.5,3)
    strvalue ='''[
     {"date":"2024-07-25" ,"value":56.5 ,"isconfidence":"False"},
     {"date":"2024-07-23" ,"value":58.5 ,"isconfidence":"False"},
     {"date":"2024-07-27" ,"value":54.5 ,"isconfidence":"False"},
     {"date":"2024-07-10" ,"value":80.5 ,"isconfidence":"False"},
     {"date":"2024-07-12" ,"value":81.5 ,"isconfidence":"False"},
     {"date":"2024-07-18" ,"value":81.5 ,"isconfidence":"False"},
     {"date":"2024-07-19" ,"value":60.5 ,"isconfidence":"False"}
    
    ]
    
    
'''
    detection.deal(strvalue)



