import math   
import random


# Radix sort in Python


# Using counting sort to sort the elements in the basis of significant places
def countingSort(array, place):
    size = len(array)
    output = [0] * size
    count = [0] * 10

    # Calculate count of elements
    for i in range(0, size):
        index = array[i] // place
        count[index % 10] += 1

    # Calculate cumulative count
    for i in range(1, 10):
        count[i] += count[i - 1]

    # Place the elements in sorted order
    i = size - 1
    while i >= 0:
        index = array[i] // place
        output[count[index % 10] - 1] = array[i]
        count[index % 10] -= 1
        i -= 1

    for i in range(0, size):
        array[i] = output[i]


# Main function to implement radix sort
def radixSort(array):
    # Get maximum element
    max_element = max(array)

    # Apply counting sort to sort elements based on place value.
    place = 1
    while max_element // place > 0:
        countingSort(array, place)
        place *= 10


data = [121, 432, 564, 23, 1, 45, 788]
radixSort(data)
print(data)











# def checkSubString(s):
#     length = len(s)
#     for i in range(int(length/2)):
#         if s[i] != s[length-1-i]:
#             return 0
#     return length

# def longestPalindrome( s):
#     """
#     :type s: str
#     :rtype: str
#     """
#     length = len(s)
    
#     if checkSubString(s)!=0:
#         return s
#     if longestPalindrome(s[:length-1])> longestPalindrome(s[1:]):
#         return s[:length-1]
#     else:
#         return s[1:]

# input ="cbbd"
# print(longestPalindrome(input))












# def containsNearbyAlmostDuplicate( nums, k, t):
#     """
#     :type nums: List[int]
#     :type k: int
#     :type t: int
#     :rtype: bool
#     """
    
#     for j in range(len(nums)):
#         # minIndx = -k+i if i>k else 0
#         # maxIndx = k+i if (k+i < len(nums) and k+i>0) else len(nums)-1
#         for i in range(i, min(len(nums), j-k)) :
#             if abs(nums[i] - nums[j]) <= t and i!=j:
#                 return True
            
#     return False 


# print(containsNearbyAlmostDuplicate([10,100,11,9,100,10],1,2))

 
# def bucketSort(array):
#     largest = max(array)
#     length = len(array)
#     size = largest/length
 
#     # Create Buckets
#     buckets = [[] for i in range(length)]
 
#     # Bucket Sorting   
#     for i in range(length):
#         index = int(array[i]/size)
#         if index != length:
#             buckets[index].append(array[i])
#         else:
#             buckets[length - 1].append(array[i])
 
#     # Sorting Individual Buckets  
#     for i in range(len(array)):
#         buckets[i] = sorted(buckets[i])
 
 
#     # Flattening the Array
#     result = []
#     for i in range(length):
#         result = result + buckets[i]
             
#     return result
 
 
# arr = [0.5, 0.4, 0.562, 10.7, 0.8888, 0.55, 0.2, 0.1, 0.4, 0.5, 0.1, 0.2]
# output = bucketSort(arr)
# print(output)


# def partition(l, r, nums):
#     # Last element will be the pivot and the first element the pointer
#     pivot, ptr = nums[r], l
#     for i in range(l, r):
#         if nums[i] <= pivot:
#             # Swapping values smaller than the pivot to the front
#             nums[i], nums[ptr] = nums[ptr], nums[i]
#             ptr += 1
#     # Finally swapping the last element with the pointer indexed number
#     nums[ptr], nums[r] = nums[r], nums[ptr]
#     return ptr
 
# # With quicksort() function, we will be utilizing the above code to obtain the pointer
# # at which the left values are all smaller than the number at pointer index and vice versa
# # for the right values.
 
 
# def quicksort(l, r, nums):
#     if len(nums) == 1:  # Terminating Condition for recursion. VERY IMPORTANT!
#         return nums
#     if l < r:
#         pi = partition(l, r, nums)
#         quicksort(l, pi-1, nums)  # Recursively sorting the left values
#         quicksort(pi+1, r, nums)  # Recursively sorting the right values
#     return nums
 
 
# example = [4, 5, 1, 2, 3,435,453,6,57,7657,5,647,635,7,354,6,5456,6345,6,4,56,\
#     46,546,65,5463,5643,5346,456,3465,4536,4563,5634,6453,574,543453,7534,32876879,8,5647,6576,675,78,8765]
# result = [1, 2, 3, 4, 5]
# print(quicksort(0, len(example)-1, example))
 

# def findMedianSortedArrays(nums1, nums1Size, nums2, nums2Size):

#     if nums1Size < nums2Size:
#         diff = nums2Size - nums1Size
#         if (diff%2 == 0):
#             diff /= 2
#             return (nums2[math.floor(diff)-1] + nums2[math.floor(diff)])/2
        
#         diff /= 2
#         return nums2[math.floor(diff)]
        
    
#     if (nums2Size < nums1Size):
#         diff= nums1Size -nums2Size
#         if (diff%2 == 0):
#             diff /= 2
#             return (nums1[-math.floor(diff)] + nums1[-math.floor(diff)-1])/2
        
#         diff /= 2
#         return nums1[- math.floor(diff) -1]
    
#     return (nums1[-1] + nums2[0])/2


# print(findMedianSortedArrays([1,2,3,4,5,6,7],7,[1,2,3,45,6,4,5,7,8,3,2,1,],12))





# from __future__ import annotations
# import numpy as np
# import pandas as pd
# from sklearn import datasets
# from IMLearn.metrics import mean_square_error
# from IMLearn.utils import split_train_test
# from IMLearn.model_selection import cross_validate
# from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
# from sklearn.linear_model import Lasso

# from utils import *
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots


# def f(x):
#     return (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)
# # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
# # and split into training- and testing portions


# X = np.random.uniform(-1.2, 2, 50)
# epsilon = np.random.normal(0, 5, 50)
# y = f(X) + epsilon
# xDF, yDF = pd.DataFrame(X), pd.Series(y)

# [xTrain, yTrain, xTest, yTest] = split_train_test(xDF, yDF, 2/3)

# xTrain = np.array(xTrain.squeeze())

# # print(xTrain)

# print(np.array(xTrain))


