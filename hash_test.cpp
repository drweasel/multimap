#include "hash_multimap.h"
#include <gtest/gtest.h>

using namespace mw;

TEST(MultiMap,Basic)
{
	hash_multimap< int,int > mp;

	mp.insert({1,2});
	mp.insert({2,3});

	EXPECT_EQ(mp.at(1),2);
	EXPECT_EQ(mp.at(2),3);

	auto i1 = mp.find(1);
	EXPECT_FALSE(i1 == mp.end());
	auto i2 = mp.find(2);
	EXPECT_FALSE(i2 == mp.end());

	auto i3 = mp.find(3);
	EXPECT_TRUE(i3 == mp.end());

	EXPECT_TRUE(mp.erase(1));
	EXPECT_TRUE(mp.erase(2));
	EXPECT_FALSE(mp.erase(3));
}

