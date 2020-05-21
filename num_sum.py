#coding=UTF8
'''
 两数之和：
给定一个整数数组 nums 和一个目标值 target，请你在该数组中找出和为目标值的那 两个 整数，并返回他们的数组下标。

你可以假设每种输入只会对应一个答案。但是，数组中同一个元素不能使用两遍。
给定 nums = [2, 7, 11, 15], target = 9

因为 nums[0] + nums[1] = 2 + 7 = 9
所以返回 [0, 1]
'''
def two_sum(nums,target):
    n = len(nums)
    res ={}
    for i in range(n):
        if target-nums[i] in res:
            return [res.get(target-nums[i]),i]
        else:
            res[nums[i]] = i
'''
总结：
    1.时间复杂度o(n)：遍历一次数组o(n)
    2.空间复杂度o(n):dict存储元素的个数
    3.技巧：字典查找的时间复杂度为o(1)
'''

'''
 三数之和：
给你一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，
使得 a + b + c = 0 ？请你找出所有满足条件且不重复的三元组。
注意：答案中不可以包含重复的三元组
示例：
给定数组 nums = [-1, 0, 1, 2, -1, -4]，

满足要求的三元组集合为：
[
  [-1, 0, 1],
  [-1, -1, 2]
]
'''
#核心思想：数组排序后双指针法
def three_sum(nums):
    lenth = len(nums)
    res = []
    if lenth < 3:
        return res
    nums.sort()
    for i in range(lenth-2):
        if nums[i] > 0:
            return res
        if i>0 and nums[i]==nums[i-1]:#跳过重复项
            continue
        cur=nums[i]
        left=i+1
        right=lenth-1
        while left < right:
            if cur+nums[left]+nums[right] == 0:
                res.append([cur,nums[left],nums[right]])
                while left<right and nums[left] == nums[left+1]:#过滤掉和左指针重复的值
                    left +=1
                while left<right and nums[right] == nums[right-1]:#过滤掉和右指针重复的值
                    right  -=1
                left += 1
                right -= 1
            elif cur+nums[left]+nums[right] < 0:
                while left<right and nums[left] == nums[left+1]:#过滤掉和左指针重复的值
                    left +=1
                left +=1
            else:
                while left<right and nums[right] == nums[right-1]:#过滤掉和右指针重复的值
                    right  -=1
                right -=1
    return res
'''
 总结：
    1.难点在处理重复项。
    2.时间复杂度O(n^2)：数组排序O(nlogn)，遍历整个数组O(n)，双指针O(n)，总的：O(nlogn)+O(n)*O(n)
    3.空间复杂度O(1)：指针使用常数大小的额外空间
'''
'''
 最接近的三数之和：
给定一个包括 n 个整数的数组 nums 和 一个目标值 target。找出 nums 中的三个整数，使得它们的和与 target 最接近。
返回这三个数的和。假定每组输入只存在唯一答案。

例如，给定数组 nums = [-1，2，1，-4], 和 target = 1.

与 target 最接近的三个数的和为 2. (-1 + 2 + 1 = 2).
'''
#个人思路：将所有三数之和都求出来，和值与对应的数组构成一个dict,最后比较和值与target差值最小即为满足条件
def threeSumClosest_1(nums,target): 
    nums.sort()
    sums ={}
    lenth = len(nums)
    for i in range(lenth):
        for j in range(i+1,lenth):
            r = lenth-1
            while r > j:
                tmp = nums[i] + nums [j] +nums[r]
                sums[tmp] = [nums[i],nums[j],nums[r]]
                r-=1
    res = [x for x in sums.keys()]
    delta = [(target-x,x) if x<target else (x-target,x) for x in res]
    delta.sort(key=lambda x:x[0])
    return delta[0][1]
#结果超出时间限制
'''
好的思路--算法流程：
特判，对于数组长度nn，如果数组为Null或者数组长度小于3，返回[]。
对数组进行排序，并定义res，保存最接近和。
遍历排序后数组：
    对于重复元素，跳过，避免重复计算（也可以不跳过）
    令左指针L=i+1,右指针R=n-1,当L<R时，执行循环：
    
    1. 令cur_sum=nums[i]+nums[L]+nums[R]cur_sum=nums[i]+nums[L]+nums[R],如果cur_sum=target返回target
    2. 若abs(cur_sum-target)<abs(res-target),说明cur_sum更接近目标，更新res
    3. 若cur_sum−target大于0，说明nums[R]太大，R左移
    4. 若cur_sum−target小于0，说明nums[L]太小，L右移
'''
#别人的思路
def threeSumClosest(nums,target):
    n=len(nums)
    if(not nums or n<3):
        return None
    nums.sort()
    res=float("inf")
    for i in range(n-2):
        if(i>0 and nums[i]==nums[i-1]):#过滤重复项
            continue
        L=i+1
        R=n-1
        while(L<R):
            cur_sum=nums[i]+nums[L]+nums[R]
            if(cur_sum==target):
                return target
            if(abs(cur_sum-target)<abs(res-target)):
                res=cur_sum
            if(cur_sum-target<0):
                L+=1
            else:
                R-=1
    return res

'''
总结： 
    1. 时间复杂度o(n^2):数组排序o(nlogn)，遍历数组o(n),双指针O(n)，总的：o(nlogn)+o(n)*o(n)
    2. 空间复杂度0(1):双指针使用常数空间
    3. 技巧：当需要固定一个，遍历其余2个的时候，可考虑双指针的办法。
'''
'''
较小的三数之和：
给定一个长度为 n 的整数数组和一个目标值 target，
* 寻找能够使条件 nums[i] + nums[j] + nums[k] < target 成立的三元组 i, j, k 个数（0 <= i < j < k < n）。
* 输入: nums = [-2,0,1,3], target = 2
* 输出: 2 
* 解释: 因为一共有两个三元组满足累加和小于 2:
     * [-2,0,1]
     * [-2,0,3]
'''
#思路：考虑先排序，再使用双指针的办法
def minThreesum(nums,target):
    nums.sort()
    n = len(nums)
    count = 0
    if not nums and n < 3:
        return count
    for i in range(n):
        l = i+1
        r = n-1
        while l < r:
            if nums[l] < target - nums[i] -nums[r]:
                count +=1
                while l < r and nums[l] == nums[l+1]:
                    l += 1
                l += 1
            elif nums[r] < target -nums[i] - nums[l]:
                count += 1
                while l < r and nums[r] == nums[r-1]:
                    r -= 1
                r -= 1
            else:
                r -= 1
    return count
#参考别人的解法：核心思想-排序后的数组，若当前三个数满足条件，那么介于r-l之间的也满足条件
def minThreesum_1(nums,target):
    nums.sort()
    n = len(nums)
    count = 0
    if not nums and n < 3:
        return count
    for i in range(n):
        l = i+1
        r = n-1
        while l < r:
            if nums[l]+ nums[i]+nums[r] < target :
                count += r-l
                l +=1
            else:
                r -= 1
    return count
'''
总结：
    1.时间复杂度o(n^2)：数组排序o(nlogn),遍历一次数组o(n),双指针遍历o(n),总的：o(nlogn)+o(n)*o(n)
    2.空间复杂度o(1):双指针使用常数的空间
    3.技巧：捕捉到关键信息---当右指针所在的三元组都满足条件的时候，由于数组是排好序的，所以：r-l之间的所有数都满足
'''
'''
盛水容器面积最大
* 给你 n 个非负整数 a1，a2，...，an，每个数代表坐标中的一个点 (i, ai) 。
* 在坐标内画 n 条垂直线，垂直线 i 的两个端点分别为 (i, ai) 和 (i, 0)。找出其中的两条线，使得它们与 x 轴共同构成的容器可以容纳最多的水。

* 说明：你不能倾斜容器，且 n 的值至少为 2。
* 输入：[1,8,6,2,5,4,8,3,7]
* 输出：49
'''
#个人思路：指针法，并用一个变量记录盛水的面积，每次和他比较，更新为较大的值
def maxArea_1(height):
    maxarea = 0
    n = len(height)
    for i in range(n):
        point = i+1
        while point <= n-1:
            if height[i] < height[point]:
                tmp = height[i] * (point-i)
            else:
                tmp = height[point] * (point-i)
            if maxarea < tmp:
                maxarea = tmp
            point +=1
    return maxarea
#结果超出时间限制：得优化--改进
#别人的思路
#双指针法，这里涉及到一个核心思想：每次移动长板不会使面积增大，移动短板有可能使面积增大，思考为什么？优化思想：缩减搜索空间！
def maxArea(height):
    res = 0
    l,r =0,len(height)-1
    while l < r:
        if height[l] < height[r]:
            res = max(res, height[l] * (r - l))
            l += 1
        else:
            res = max(res, height[r] * (r - l))
            r -= 1
    return res
'''
总结：
    1. 时间复杂度o(n)：只需要循环一次
    2. 空间复杂度o(1):双指针使用常数空间
    3. 技巧：缩减搜索空间--状态消去法
'''