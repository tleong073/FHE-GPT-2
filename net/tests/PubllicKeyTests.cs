﻿using Microsoft.Research.SEAL;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace SEALNetTest
{
    [TestClass]
    public class PubllicKeyTests
    {
        [TestMethod]
        public void SaveLoadTest()
        {
            List<SmallModulus> coeffModulus = new List<SmallModulus>
            {
                DefaultParams.SmallMods40Bit(0)
            };
            EncryptionParameters parms = new EncryptionParameters(SchemeType.BFV)
            {
                PolyModulusDegree = 64,
                PlainModulus = new SmallModulus(1 << 6),
                CoeffModulus = coeffModulus
            };
            SEALContext context = SEALContext.Create(parms);
            KeyGenerator keygen = new KeyGenerator(context);

            PublicKey pub = keygen.PublicKey;

            Assert.IsNotNull(pub);
            Assert.AreEqual(2, pub.Data.Size);
            Assert.IsTrue(pub.Data.IsNTTForm);

            PublicKey pub2 = new PublicKey();
            MemoryPoolHandle handle = pub2.Pool;

            Assert.AreEqual(0, pub2.Data.Size);
            Assert.IsFalse(pub2.Data.IsNTTForm);
            Assert.AreEqual(ParmsId.Zero, pub2.ParmsId);

            using (MemoryStream stream = new MemoryStream())
            {
                pub.Save(stream);

                stream.Seek(offset: 0, loc: SeekOrigin.Begin);

                pub2.Load(context, stream);
            }

            Assert.AreNotSame(pub, pub2);
            Assert.AreEqual(2, pub2.Data.Size);
            Assert.IsTrue(pub2.Data.IsNTTForm);
            Assert.AreEqual(pub.ParmsId, pub2.ParmsId);
            Assert.AreNotEqual(ParmsId.Zero, pub2.ParmsId);
            Assert.IsTrue(handle.AllocByteCount != 0ul);
        }
    }
}