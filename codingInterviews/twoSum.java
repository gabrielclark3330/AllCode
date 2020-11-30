class Solution {

    public static void main(String[] args)
    {
        int[] example = {2,7,11,15};
        int exTarget = 9;
        System.out.println(twoSum(example, exTarget));
    }

    public static int[] twoSum(int[] nums, int target) {
        int[] example = {2,7,11,15};
        int exTarget = 9;
        int[] output;
        
        for (i = 0, i < example.length, i++){
            for (j = 0, j<arr.length, j++){
                if (j+i == exTarget){
                    output = Arrays.copyOf(example, example.length + 1);
                    output[arr.length - 1] = j;
                    output = Arrays.copyOf(example, example.length + 1);
                    output[arr.length - 1] = i;
                }
            }
        }

        return output;
    }
}