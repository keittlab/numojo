from testing import *
from numojo.xoshiro import *


# Comparison values generated by xoroshiro128p.c
def test_xoshiro256plus():
    var rng = Xoshiro256Plus(123456789)
    assert_equal(rng.next(), 12059086120572499294)
    rng.jump()
    assert_equal(rng.next(), 6473535184725534191)
    rng.long_jump()
    assert_equal(rng.next(), 12794950360312403446)


def test_xoshiro256plusplus():
    var rng = Xoshiro256PlusPlus(123456789)
    assert_equal(rng.next(), 11089759438045651894)
    rng.jump()
    assert_equal(rng.next(), 66056455977260917)
    rng.long_jump()
    assert_equal(rng.next(), 3938992559361620932)


def test_xoshiro256starstar():
    var rng = Xoshiro256StarStar(123456789)
    assert_equal(rng.next(), 15127205273500847298)
    rng.jump()
    assert_equal(rng.next(), 14782342367199966867)
    rng.long_jump()
    assert_equal(rng.next(), 4400463616481334238)


def test_xoshiro256ParallelPlusPlus():
    var seed: UInt64 = 123
    var rng_par = Xoshiro256PlusPlusSIMD[4](seed)
    var rng1 = Xoshiro256PlusPlus(seed)
    var rng2 = Xoshiro256PlusPlus(seed)
    var rng3 = Xoshiro256PlusPlus(seed)
    var rng4 = Xoshiro256PlusPlus(seed)
    rng2.long_jump()
    rng3.long_jump()
    rng3.long_jump()
    rng4.long_jump()
    rng4.long_jump()
    rng4.long_jump()
    rng_par.step()
    rng1.step()
    rng2.step()
    rng3.step()
    rng4.step()
    rng_par.step()
    rng1.step()
    rng2.step()
    rng3.step()
    rng4.step()
    assert_equal(rng_par.s0[0], rng1.s0)
    assert_equal(rng_par.s1[0], rng1.s1)
    assert_equal(rng_par.s2[0], rng1.s2)
    assert_equal(rng_par.s3[0], rng1.s3)
    assert_equal(rng_par.s0[1], rng2.s0)
    assert_equal(rng_par.s1[1], rng2.s1)
    assert_equal(rng_par.s2[1], rng2.s2)
    assert_equal(rng_par.s3[1], rng2.s3)
    assert_equal(rng_par.s0[2], rng3.s0)
    assert_equal(rng_par.s1[2], rng3.s1)
    assert_equal(rng_par.s2[2], rng3.s2)
    assert_equal(rng_par.s3[2], rng3.s3)
    assert_equal(rng_par.s0[3], rng4.s0)
    assert_equal(rng_par.s1[3], rng4.s1)
    assert_equal(rng_par.s2[3], rng4.s2)
    assert_equal(rng_par.s3[3], rng4.s3)
    rng_par.step()
    rng1.step()
    rng2.step()
    rng3.step()
    rng4.step()
    assert_equal(rng_par.s0[0], rng1.s0)
    assert_equal(rng_par.s1[0], rng1.s1)
    assert_equal(rng_par.s2[0], rng1.s2)
    assert_equal(rng_par.s3[0], rng1.s3)
    assert_equal(rng_par.s0[1], rng2.s0)
    assert_equal(rng_par.s1[1], rng2.s1)
    assert_equal(rng_par.s2[1], rng2.s2)
    assert_equal(rng_par.s3[1], rng2.s3)
    assert_equal(rng_par.s0[2], rng3.s0)
    assert_equal(rng_par.s1[2], rng3.s1)
    assert_equal(rng_par.s2[2], rng3.s2)
    assert_equal(rng_par.s3[2], rng3.s3)
    assert_equal(rng_par.s0[3], rng4.s0)
    assert_equal(rng_par.s1[3], rng4.s1)
    assert_equal(rng_par.s2[3], rng4.s2)
    assert_equal(rng_par.s3[3], rng4.s3)
