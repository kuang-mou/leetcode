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